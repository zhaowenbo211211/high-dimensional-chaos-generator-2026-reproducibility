// Maintainer: zhaowenbo
#ifndef PRNG371_H
#define PRNG371_H

#include <stdint.h>

/* PRNG-371 bit-exact C implementation
 * Based on hw/src_371/prng_top.cpp (HLS model)
 *
 * Parameters (from chaos_params.h):
 *   A_J1_0 = 371, A_J1_1 = 373, A_J1_2 = 375
 *   A_JN_0 =  50, A_JN_1 =  48, A_JN_2 =  46
 *   B_VAL  = 111
 *
 * State: 6 x 16-bit (x1,x2,x3,s0,s1,s2) where s0=x4
 * Output: 64-bit mixed word y_t (little-endian byte stream)
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize PRNG-371 with 4 x 16-bit seeds.
 * If all seeds are 0, defaults [1000,2000,3000,4000] are used.
 * Warmup of 2000 iterations is performed internally.
 */
void prng371_init(uint16_t x1, uint16_t x2, uint16_t x3, uint16_t x4);

/* Generate the next 64-bit PRNG word (post-mixer output y_t). */
uint64_t prng371_next(void);

/* Generate n_bytes of keystream into buf (little-endian per 64-bit word). */
void prng371_generate_bytes(uint8_t *buf, uint16_t n_bytes);

/* Convenience: encrypt/decrypt in-place with current keystream.
 * Call prng371_init() with the SAME seed on both sides before using.
 */
void prng371_crypt(uint8_t *data, uint16_t len);

#ifdef __cplusplus
}
#endif

#endif /* PRNG371_H */
