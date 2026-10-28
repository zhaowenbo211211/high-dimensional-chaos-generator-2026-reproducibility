#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Chaos-based PRNG engine.

Combines the enhanced 4D chaotic map (HighDimChaosMapEnhanced) with
the lightweight ROTL-XOR word-level post-processor.

Architecture:
  x^{(t)} = G_enhanced(x^{(t-1)})               # enhanced 4D chaos
  u_t     = x1 || x2 || x3 || x4                 # concatenation
  y_t     = u_t ⊕ ROTL64(u_{t-1}, 17) ⊕ ROTL64(u_{t-2}, 43)  # mixing
  output  = y_t                                   # 64-bit PRNG block
"""

from typing import Dict, Optional
import numpy as np

from chaos.core.params import ComplementaryChaosParams
from chaos.core.enhanced_map import HighDimChaosMapEnhanced
from chaos.application.prng.post_processor import PostProcessor


class ChaosPRNG:
    """64-bit PRNG driven by enhanced 4D chaos with ROTL-XOR post-processing.

    Parameters
    ----------
    params : Dict
        Chaotic map parameters from ComplementaryChaosParams.synthesize().
    seed : int or None
        Random seed for the chaotic map.
    word_size : int
        Word size in bits (default 16 → 4×16 = 64-bit output).
    """

    def __init__(self, params: Dict, seed: Optional[int] = None,
                 word_size: int = 16, warmup: int = 2000):
        if len(params['a_j1']) + 1 != 4:
            raise ValueError("ChaosPRNG requires n=4 chaotic map")

        self._map = HighDimChaosMapEnhanced(params, word_size=word_size,
                                            seed=seed)
        self._post = PostProcessor(word_bits=word_size)
        self.word_size = word_size
        self._warmup_count = warmup
        # Note: warmup is NOT done in __init__ — call set_state() or
        # seed_from_key() first, which will trigger warmup automatically.

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def next(self) -> int:
        """Advance one step and return a 64-bit mixed output word."""
        state = self._map.next()
        return self._post.mix(state)

    def generate(self, n: int,
                 progress_interval: int = 0) -> np.ndarray:
        """Generate n consecutive 64-bit PRNG output words.

        Parameters
        ----------
        n : int  number of 64-bit words to generate.
        progress_interval : int  if > 0, print progress every N words and
            flush stdout (also serves as Ctrl+C interrupt checkpoint).

        Returns
        -------
        np.ndarray  shape (n,), dtype=np.uint64
        """
        outputs = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            outputs[i] = self.next()
            if progress_interval > 0 and (i + 1) % progress_interval == 0:
                pct = 100.0 * (i + 1) / n
                print(f"    progress: {i+1:,}/{n:,} words ({pct:.1f}%)",
                      flush=True)
        return outputs

    def generate_bytes(self, n_bytes: int) -> bytes:
        """Generate n_bytes of PRNG output.

        Parameters
        ----------
        n_bytes : int  number of bytes to generate

        Returns
        -------
        bytes
        """
        words_needed = (n_bytes + 7) // 8
        words = self.generate(words_needed)
        return words.tobytes()[:n_bytes]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_warmup(self) -> None:
        """Discard the first _warmup_count outputs to eliminate transients."""
        for _ in range(self._warmup_count):
            self.next()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def seed_from_key(self, key: int) -> None:
        """Initialize PRNG state from a single integer key.

        Derives a deterministic 4D initial state from the key by hashing it
        through numpy's random generator, then resets the post-processor.
        Same key always produces the same output stream.

        Parameters
        ----------
        key : int  arbitrary integer key (e.g., 0 ≤ key < 2^64).
        """
        # Mix the key bytes into numpy's PRNG to derive initial state
        key_bytes = key.to_bytes((key.bit_length() + 7) // 8 + 1, 'little')
        seed_hash = int.from_bytes(key_bytes, 'little') % (2**31 - 1)
        np.random.seed(seed_hash)
        init_state = np.random.randint(0, 1 << self.word_size, size=4,
                                       dtype=np.int64)
        self._map.set_state_from_physical(init_state)
        # Clear delay-chain entries (s_1 .. s_{n-2}) to avoid state leakage
        if self._map.state_dim > self._map.n:
            self._map.state[self._map.n:] = 0
        self._post.reset()
        self._map.iter_count = 0
        self._do_warmup()

    def set_state(self, x1: int, x2: int, x3: int, x4: int) -> None:
        """Set the internal 4D chaotic state directly, then warm-up.

        If all four values are 0, defaults [1000, 2000, 3000, 4000] are used,
        matching the HLS hardware behaviour.
        """
        if x1 == 0 and x2 == 0 and x3 == 0 and x4 == 0:
            x1, x2, x3, x4 = 1000, 2000, 3000, 4000
        phys = np.array([x1, x2, x3, x4], dtype=np.int64)
        self._map.set_state_from_physical(phys)
        if self._map.state_dim > self._map.n:
            self._map.state[self._map.n:] = 0
        self._post.reset()
        self._map.iter_count = 0
        self._do_warmup()

    def get_state(self) -> np.ndarray:
        """Return current 4D physical state [x1, x2, x3, x4]."""
        return self._map.get_physical_state()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset chaotic map state and post-processor. Call set_state() after."""
        self._map.reset(seed=seed)
        self._post.reset()

    # ------------------------------------------------------------------
    # Binary sequence export
    # ------------------------------------------------------------------

    def export_binary_sequence(self, filepath: str,
                                length: int = 100_000_000,
                                mode: str = "raw") -> None:
        """Generate a binary sequence and export to file.

        Parameters
        ----------
        filepath : str    output file path.
        length : int      number of bits to generate (default 10^8).
        mode : str        output format:
            - "raw"   : pack 8 bits per byte, write as binary file.
            - "ascii" : write '0' and '1' characters (1 byte per bit).
        """
        if mode not in ("raw", "ascii"):
            raise ValueError(f"mode must be 'raw' or 'ascii', got '{mode}'")

        # Each 64-bit PRNG word provides 64 bits
        words_needed = (length + 63) // 64
        words = self.generate(words_needed,
                              progress_interval=max(1, words_needed // 10))

        if mode == "raw":
            self._export_raw(filepath, words, length)
        else:
            self._export_ascii(filepath, words, length)

    def _export_raw(self, filepath: str, words: np.ndarray,
                    length: int) -> None:
        """Pack bits from words into bytes and write binary file."""
        total_bytes = (length + 7) // 8
        raw = np.zeros(total_bytes, dtype=np.uint8)

        bit_idx = 0
        for w in words:
            for b in range(64):
                if bit_idx >= length:
                    break
                byte_pos = bit_idx // 8
                bit_pos = bit_idx % 8
                raw[byte_pos] |= ((int(w) >> b) & 1) << bit_pos
                bit_idx += 1
            if bit_idx >= length:
                break

        with open(filepath, "wb") as f:
            f.write(raw.tobytes())
        print(f"Binary sequence written to: {filepath}")
        print(f"  length = {length} bits ({total_bytes} bytes)")
        print(f"  mode   = raw")

    def _export_ascii(self, filepath: str, words: np.ndarray,
                      length: int) -> None:
        """Write bits as '0'/'1' ASCII characters."""
        chunk_size = 1_000_000  # write in 1 MB chunks
        bit_idx = 0

        with open(filepath, "w") as f:
            for w in words:
                if bit_idx >= length:
                    break
                # Extract bits LSB-first from this word
                remaining = min(64, length - bit_idx)
                chars = []
                for b in range(remaining):
                    chars.append('1' if ((int(w) >> b) & 1) else '0')
                f.write(''.join(chars))
                bit_idx += remaining

                # Progress indicator
                if bit_idx % (10_000_000) < 64:
                    pct = 100.0 * bit_idx / length
                    print(f"  exporting ascii: {bit_idx:,}/{length:,} bits "
                          f"({pct:.1f}%)")

        print(f"ASCII sequence written to: {filepath}")
        print(f"  length = {length:,} bits ({length:,} bytes)")
        print(f"  mode   = ascii")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_r1_ly(cls, A_odd: int = 271, B_even: int = 50,
                   seed: Optional[int] = None,
                   word_size: int = 16,
                   warmup: int = 2000) -> "ChaosPRNG":
        """Factory: create a ChaosPRNG from R1-LY synthesis parameters.

        Parameters
        ----------
        A_odd : int  odd base gain (default 271).
        B_even : int  even base gain (default 50).
        seed : int or None
        word_size : int  default 16.
        warmup : int  transient outputs to discard (default 1000).

        Returns
        -------
        ChaosPRNG
        """
        synth = ComplementaryChaosParams(n=4, A_odd=A_odd, B_even=B_even)
        params = synth.synthesize()
        return cls(params, seed=seed, word_size=word_size, warmup=warmup)
