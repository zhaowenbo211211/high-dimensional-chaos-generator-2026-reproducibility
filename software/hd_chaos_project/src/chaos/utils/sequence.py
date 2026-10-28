#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""NIST test sequence generation from chaotic map output."""

import numpy as np
from chaos.core.base_map import HighDimChaosMap


def generate_binary_sequence(map_instance: HighDimChaosMap,
                             length: int = 1_000_000) -> bytes:
    """Generate a binary test sequence for NIST SP 800-22 testing.

    Iterates the map, extracts the lowest 16 bits of x_n, and packs into bytes.
    """
    samples_needed = (length + 15) // 16
    data = map_instance.iterate(samples_needed + 1000)[1000:]

    bits = np.zeros(length, dtype=np.uint8)
    for i, value in enumerate(data[:samples_needed, -1]):
        for bit in range(16):
            idx = i * 16 + bit
            if idx >= length:
                break
            bits[idx] = (value >> bit) & 1

    return np.packbits(bits).tobytes()
