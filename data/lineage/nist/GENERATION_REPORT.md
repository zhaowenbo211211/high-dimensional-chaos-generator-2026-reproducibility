# PRNG-371 NIST Input Generation Report

- Variant: `chaos_prng_371` / PRNG-371
- Warmup: 2000
- Seed: [1000, 2000, 3000, 4000]
- Num sequences: 100
- Bits per sequence: 1000000
- Total bits: 100000000
- Format selected for STS: `ascii_bits`
- Selected output: `prng_experiment/randomness/prng371_warmup2000/input/prng371_warmup2000_100x1000000.asc`
- Binary output: `prng_experiment/randomness/prng371_warmup2000/input/prng371_warmup2000_100x1000000.bin`
- ASCII output: `prng_experiment/randomness/prng371_warmup2000/input/prng371_warmup2000_100x1000000.asc`
- Selected output SHA-256: `0952f43ed4901b70fcdf6a01df90e552bfbf6ad4b532a09a32e8fba126ed87b4`
- Binary SHA-256: `7a8f9c4f411363acd878b6cb2008719267e2b29bb87fba625e4d10bc43ebab97`
- ASCII SHA-256: `0952f43ed4901b70fcdf6a01df90e552bfbf6ad4b532a09a32e8fba126ed87b4`
- Byte order: little-endian per 64-bit output word.
- Source: PC golden PRNG-371 model (`hw/src_371/golden_prng.py` + `golden_371.so`).
- Golden reference alignment match: `False`

This is PC golden PRNG-371 output. It is not FPGA physical validation.
