// Maintainer: zhaowenbo
/*
 * PRNG-371 UDP Image Encryption Sensor Node
 *
 * Uses a pre-generated 128x128 grayscale portrait image (image_data.h).
 * Shared parameters (known to both sides before transmission):
 *   - Image: 128x128 grayscale = 16384 bytes
 *   - Chunk: 64 bytes  ->  256 chunks total
 *   - Seed:  [1000, 2000, 3000, 4000]
 *   - Warmup: 2000
 */

#include "contiki.h"
#include "net/ipv6/simple-udp.h"
#include "sys/log.h"
#include <stdint.h>
#include <string.h>

#define LOG_MODULE "Sensor"
#define LOG_LEVEL LOG_LEVEL_INFO

#include "prng371.h"
#include "image_data.h"

#define UDP_SENSOR_PORT  8765
#define UDP_GATEWAY_PORT 5678

/* Image parameters from image_data.h */
#define CHUNK_SIZE  64
#define NUM_CHUNKS  (IMG_SIZE / CHUNK_SIZE)    /* 256 */

static const uint16_t prng_seed[4] = {1000, 2000, 3000, 4000};

static struct simple_udp_connection udp_conn;
static uint8_t chunk_buf[1 + CHUNK_SIZE];

/*---------------------------------------------------------------------------*/
PROCESS(sensor_process, "PRNG-371 Sensor Node");
AUTOSTART_PROCESSES(&sensor_process);
/*---------------------------------------------------------------------------*/

PROCESS_THREAD(sensor_process, ev, data)
{
  static struct etimer timer;
  static uip_ipaddr_t dest_ipaddr;
  static int chunk_idx;
  static int i;
  static uint8_t keystream[CHUNK_SIZE];

  PROCESS_BEGIN();

  simple_udp_register(&udp_conn, UDP_SENSOR_PORT, NULL,
                      UDP_GATEWAY_PORT, NULL);

  LOG_INFO("Sensor: portrait image ready (%dx%d = %d bytes, %d chunks of %d bytes)\n",
           IMG_WIDTH, IMG_HEIGHT, IMG_SIZE, NUM_CHUNKS, CHUNK_SIZE);

  /* Wait for network setup and gateway to be ready */
  etimer_set(&timer, CLOCK_SECOND * 2);
  PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&timer));

  /* Initialize PRNG-371 */
  prng371_init(prng_seed[0], prng_seed[1], prng_seed[2], prng_seed[3]);
  LOG_INFO("Sensor: PRNG-371 ready, seed=[%u,%u,%u,%u]\n",
           prng_seed[0], prng_seed[1], prng_seed[2], prng_seed[3]);

  /* Build destination: link-local all-nodes multicast */
  uip_create_linklocal_allnodes_mcast(&dest_ipaddr);

  LOG_INFO("Sensor: === STARTING TRANSMISSION ===\n");

  /* Send encrypted chunks one by one (stream cipher) */
  for (chunk_idx = 0; chunk_idx < NUM_CHUNKS; chunk_idx++) {
    /* Generate keystream and encrypt THIS chunk */
    prng371_generate_bytes(keystream, CHUNK_SIZE);
    chunk_buf[0] = (uint8_t)chunk_idx;
    for (i = 0; i < CHUNK_SIZE; i++) {
      chunk_buf[1 + i] = image_data[chunk_idx * CHUNK_SIZE + i] ^ keystream[i];
    }

    simple_udp_sendto(&udp_conn, chunk_buf, 1 + CHUNK_SIZE, &dest_ipaddr);
    /* LOG_INFO("Sensor: chunk %3d/%d encrypted and sent\n", chunk_idx + 1, NUM_CHUNKS); */

    /* Fast pacing: 50 ms between packets */
    etimer_set(&timer, CLOCK_SECOND / 20);
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&timer));
  }

  LOG_INFO("Sensor: === TRANSMISSION COMPLETE ===\n");

  PROCESS_END();
}
/*---------------------------------------------------------------------------*/
