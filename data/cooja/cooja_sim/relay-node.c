// Maintainer: zhaowenbo
/*
 * Relay Node: receives image part from Gateway via UDP, then exports hex.
 *
 * Each relay listens on a unique port (9000 + node_id - 3).
 * Packet format: [seq:1][data:64]
 * Redundant transmissions handled by dedup (seq).
 */

#include "contiki.h"
#include "net/ipv6/simple-udp.h"
#include "sys/node-id.h"
#include "sys/log.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define LOG_MODULE "Relay"
#define LOG_LEVEL LOG_LEVEL_INFO

#define RELAY_PORT_BASE 9000
#define PART_SIZE       2048
#define PKT_PAYLOAD     64
#define MAX_PKTS        (PART_SIZE / PKT_PAYLOAD)   /* 32 */
#define EXPECTED_PKTS   MAX_PKTS

static struct simple_udp_connection udp_conn;
static uint8_t rx_buffer[PART_SIZE];
static uint8_t rx_bitmap[MAX_PKTS];
static uint16_t rx_count = 0;

/*---------------------------------------------------------------------------*/
PROCESS(relay_process, "Relay Node");
AUTOSTART_PROCESSES(&relay_process);
/*---------------------------------------------------------------------------*/

static void export_part_hex(void)
{
  int i, j, pos;
  char line[120];

  LOG_INFO("Relay: >>>BEGIN_PART<<<\n");

  for (i = 0; i < PART_SIZE; i += 48) {
    pos = snprintf(line, sizeof(line), "R%03d:", i / 48);
    for (j = 0; j < 48 && (i + j) < PART_SIZE && pos < (int)sizeof(line) - 3; j++) {
      pos += snprintf(line + pos, sizeof(line) - pos, "%02x", rx_buffer[i + j]);
    }
    LOG_INFO("Relay: %s\n", line);
  }

  LOG_INFO("Relay: >>>END_PART<<<\n");
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
  uint8_t seq;
  uint16_t payload_len;

  if (datalen < 2) return;

  seq = data[0];
  if (seq >= MAX_PKTS) return;

  payload_len = datalen - 1;
  if (seq * PKT_PAYLOAD + payload_len > PART_SIZE) {
    payload_len = PART_SIZE - seq * PKT_PAYLOAD;
  }

  if (!rx_bitmap[seq]) {
    memcpy(rx_buffer + seq * PKT_PAYLOAD, &data[1], payload_len);
    rx_bitmap[seq] = 1;
    rx_count++;
  }

  if (rx_count >= EXPECTED_PKTS) {
    process_poll(&relay_process);
  }
}

/*---------------------------------------------------------------------------*/
PROCESS_THREAD(relay_process, ev, data)
{
  static struct etimer timeout;
  uint8_t my_id;
  uint16_t my_port;

  PROCESS_BEGIN();

  my_id = node_id;
  my_port = RELAY_PORT_BASE + (my_id - 3);

  simple_udp_register(&udp_conn, my_port, NULL, 0, udp_rx_callback);

  LOG_INFO("Relay: ID=%u listening on port %u\n", my_id, my_port);

  etimer_set(&timeout, CLOCK_SECOND * 60);
  while (rx_count < EXPECTED_PKTS) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&timeout) || rx_count == EXPECTED_PKTS);
    if (etimer_expired(&timeout)) break;
  }

  if (rx_count == EXPECTED_PKTS) {
    LOG_INFO("Relay: all %d packets received, exporting...\n", EXPECTED_PKTS);
  } else {
    LOG_INFO("Relay: TIMEOUT, received %d/%d packets\n", rx_count, EXPECTED_PKTS);
  }

  export_part_hex();
  LOG_INFO("Relay: === DONE ===\n");

  PROCESS_END();
}
/*---------------------------------------------------------------------------*/
