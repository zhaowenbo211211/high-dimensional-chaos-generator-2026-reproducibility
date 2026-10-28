// Maintainer: zhaowenbo
/*
 * PRNG-371 UDP Image Decryption Gateway Node
 *
 * Shared parameters (known before transmission):
 *   - Image: 128x128 grayscale = 16384 bytes
 *   - Chunk: 64 bytes  ->  256 chunks total
 *   - Seed:  [1000, 2000, 3000, 4000]
 *   - Warmup: 2000
 *
 * Behavior:
 *   1. Buffer ALL received ciphertext chunks (do NOT decrypt yet).
 *   2. Once ALL chunks received OR timeout (20s) expires:
 *      - Decrypt entire image in one sequential pass.
 *      - Lost chunks are filled with 0x00.
 *   3. Report bitmap, match count, and lost chunks.
 *   4. Distribute decrypted image to 8 relays SEQUENTIALLY using etimer
 *      (clock_delay does NOT advance simulation time in Cooja).
 */

#include "contiki.h"
#include "net/ipv6/simple-udp.h"
#include "sys/log.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define LOG_MODULE "Gateway"
#define LOG_LEVEL LOG_LEVEL_INFO

#include "prng371.h"
#include "image_data.h"

#define UDP_SENSOR_PORT  8765
#define UDP_GATEWAY_PORT 5678

#define MAX_CHUNKS      300
#define MAX_CHUNK_SIZE  64

/* Shared parameters */
#define IMG_WIDTH       128
#define IMG_HEIGHT      128
#define IMG_SIZE        (IMG_WIDTH * IMG_HEIGHT)
#define CHUNK_SIZE      64
#define NUM_CHUNKS      (IMG_SIZE / CHUNK_SIZE)
#define RX_TIMEOUT_SEC  20

/* Distribution parameters */
#define NUM_RELAYS      8
#define RELAY_BYTES     2048
#define PKT_PAYLOAD     64
#define PKTS_PER_RELAY  (RELAY_BYTES / PKT_PAYLOAD)   /* 32 */

static struct simple_udp_connection udp_conn;
static const uint16_t prng_seed[4] = {1000, 2000, 3000, 4000};

/* Reception state */
static uint8_t rx_bitmap[NUM_CHUNKS];
static uint8_t rx_cipher[NUM_CHUNKS][MAX_CHUNK_SIZE];
static uint16_t rx_count = 0;
static uint8_t done = 0;

/* Decrypted output */
static uint8_t decrypted_image[IMG_SIZE];

/*---------------------------------------------------------------------------*/
PROCESS(gateway_process, "PRNG-371 Gateway Node");
AUTOSTART_PROCESSES(&gateway_process);
/*---------------------------------------------------------------------------*/

static void decrypt_and_reconstruct(void)
{
  int chunk_idx, i;
  uint8_t keystream[CHUNK_SIZE];

  LOG_INFO("Gateway: --- DECRYPT & RECONSTRUCT ---\n");

  prng371_init(prng_seed[0], prng_seed[1], prng_seed[2], prng_seed[3]);

  for (chunk_idx = 0; chunk_idx < NUM_CHUNKS; chunk_idx++) {
    prng371_generate_bytes(keystream, CHUNK_SIZE);

    if (rx_bitmap[chunk_idx]) {
      for (i = 0; i < CHUNK_SIZE; i++) {
        decrypted_image[chunk_idx * CHUNK_SIZE + i] =
          rx_cipher[chunk_idx][i] ^ keystream[i];
      }
    } else {
      for (i = 0; i < CHUNK_SIZE; i++) {
        decrypted_image[chunk_idx * CHUNK_SIZE + i] = 0x00;
      }
    }
  }
}

static void verify_and_report(void)
{
  int i;
  int match = 0, mismatch = 0, lost = 0;
  uint8_t expected;
  int lost_chunks = 0;

  LOG_INFO("Gateway: --- VERIFICATION ---\n");

  LOG_INFO("Gateway: bitmap = ");
  for (i = 0; i < NUM_CHUNKS; i++) {
    LOG_INFO_("%c", rx_bitmap[i] ? '1' : '0');
    if (!rx_bitmap[i]) lost_chunks++;
  }
  LOG_INFO_(" (received %d/%d, lost %d chunks)\n", rx_count, NUM_CHUNKS, lost_chunks);

  for (i = 0; i < IMG_SIZE; i++) {
    expected = image_data[i];

    int chunk_idx = i / CHUNK_SIZE;
    if (!rx_bitmap[chunk_idx]) {
      lost++;
    } else if (decrypted_image[i] == expected) {
      match++;
    } else {
      mismatch++;
    }
  }

  LOG_INFO("Gateway: pixels  match=%d mismatch=%d lost=%d\n", match, mismatch, lost);

  if (mismatch == 0 && lost == 0) {
    LOG_INFO("Gateway: >>> PERFECT RECOVERY (MSE=0) <<<\n");
  } else if (mismatch == 0 && lost > 0) {
    LOG_INFO("Gateway: >>> PARTIAL RECOVERY: %.1f%% received <<<\n",
             (100.0 * match) / IMG_SIZE);
  } else {
    LOG_INFO("Gateway: >>> DECRYPTION ERROR <<<\n");
  }
}

/*---------------------------------------------------------------------------*/
static void
udp_rx_callback(struct simple_udp_connection *c,
                const uip_ipaddr_t *sender_addr,
                uint16_t sender_port,
                const uip_ipaddr_t *receiver_addr,
                uint16_t receiver_port,
                const uint8_t *data,
                uint16_t datalen)
{
  uint8_t idx;

  if (datalen == 0 || done) return;

  idx = data[0];

  if (idx >= NUM_CHUNKS) {
    LOG_WARN("Gateway: invalid chunk index %u\n", idx);
    return;
  }

  if (rx_bitmap[idx]) {
    LOG_INFO("Gateway: duplicate chunk %u ignored\n", idx);
    return;
  }

  rx_bitmap[idx] = 1;
  rx_count++;
  if (datalen > 1) {
    memcpy(rx_cipher[idx], &data[1], datalen - 1);
  }

  if (rx_count == NUM_CHUNKS) {
    process_poll(&gateway_process);
  }
}
/*---------------------------------------------------------------------------*/

PROCESS_THREAD(gateway_process, ev, data)
{
  static struct etimer timeout;
  static struct etimer send_timer;
  static uip_ipaddr_t dest_addr;
  static uint8_t buf[1 + PKT_PAYLOAD];
  static int current_relay, pkt, copy;
  static int offset, payload_len;
  static uint16_t port;

  PROCESS_BEGIN();

  simple_udp_register(&udp_conn, UDP_GATEWAY_PORT, NULL,
                      UDP_SENSOR_PORT, udp_rx_callback);

  LOG_INFO("Gateway: ready, expecting %d chunks of %d bytes each\n",
           NUM_CHUNKS, CHUNK_SIZE);
  LOG_INFO("Gateway: timeout = %ds\n", RX_TIMEOUT_SEC);

  etimer_set(&timeout, CLOCK_SECOND * RX_TIMEOUT_SEC);

  while (!done) {
    PROCESS_WAIT_EVENT();

    if (ev == PROCESS_EVENT_POLL) {
      LOG_INFO("Gateway: ALL CHUNKS RECEIVED early!\n");
    } else if (etimer_expired(&timeout)) {
      LOG_INFO("Gateway: TIMEOUT (%ds) - processing received chunks\n", RX_TIMEOUT_SEC);
    } else {
      continue;
    }

    done = 1;
    decrypt_and_reconstruct();
    verify_and_report();
    break;
  }

  /* --- DISTRIBUTION PHASE --- */
  LOG_INFO("Gateway: --- DISTRIBUTE TO RELAYS (SEQUENTIAL) ---\n");

  /* Use multicast address - each relay isolates by its own UDP port */
  uip_create_linklocal_allnodes_mcast(&dest_addr);

  for (current_relay = 0; current_relay < NUM_RELAYS; current_relay++) {
    port = 9000 + current_relay;

    for (pkt = 0; pkt < PKTS_PER_RELAY; pkt++) {
      offset = current_relay * RELAY_BYTES + pkt * PKT_PAYLOAD;
      payload_len = PKT_PAYLOAD;
      if (offset + payload_len > IMG_SIZE) {
        payload_len = IMG_SIZE - offset;
      }

      buf[0] = (uint8_t)pkt;
      memcpy(&buf[1], &decrypted_image[offset], payload_len);

      /* Send 2 copies per packet with ~30ms spacing */
      for (copy = 0; copy < 2; copy++) {
        simple_udp_sendto_port(&udp_conn, buf, 1 + payload_len,
                               &dest_addr, port);
        if (copy < 1) {
          etimer_set(&send_timer, CLOCK_SECOND / 30);
          PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&send_timer));
        }
      }
      /* ~50ms between packets */
      etimer_set(&send_timer, CLOCK_SECOND / 20);
      PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&send_timer));
    }

    LOG_INFO("Gateway: part %d/%d sent to port %u\n",
             current_relay + 1, NUM_RELAYS, port);

    /* Wait 2 seconds before starting next relay.
     * This gives the current relay time to receive & export.
     * CRITICAL: use etimer (yields to Cooja scheduler) instead of
     * clock_delay (busy-wait, does NOT advance simulation time). */
    etimer_set(&send_timer, CLOCK_SECOND * 2);
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&send_timer));
  }

  LOG_INFO("Gateway: === DISTRIBUTION COMPLETE ===\n");
  LOG_INFO("Gateway: === SIMULATION COMPLETE ===\n");

  PROCESS_END();
}
/*---------------------------------------------------------------------------*/
