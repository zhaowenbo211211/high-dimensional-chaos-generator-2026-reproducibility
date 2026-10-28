#!/usr/bin/env python3
# Maintainer: zhaowenbo
"""
Generate paper figures for PRNG-371 Cooja simulation:
  1. original.bmp      — from image_data.h
  2. encrypted.bmp     — PRNG-371 encrypted
  3. cooja_decrypted.png — copy from simulation result
  4. README.md         — figure descriptions
"""

import ctypes
import os
import re
import shutil
import struct

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
PAPER_FIGURES = os.path.join(PROJECT_ROOT, "paper_figures")
IMAGE_DATA_H = os.path.join(SCRIPT_DIR, "image_data.h")
COOJA_DECRYPTED = os.path.join(SCRIPT_DIR, "cooja_decrypted.png")
LIBPRNG = os.path.join(SCRIPT_DIR, "libprng371.so")

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_SIZE = IMG_WIDTH * IMG_HEIGHT

# ---------------------------------------------------------------------------
# 1. Parse image_data.h
# ---------------------------------------------------------------------------
def parse_image_data():
    with open(IMAGE_DATA_H, "r") as f:
        content = f.read()
    # Find array contents between braces
    m = re.search(r"image_data\[IMG_SIZE\]\s*=\s*\{(.+?)\};", content, re.DOTALL)
    if not m:
        raise RuntimeError("Could not find image_data array in image_data.h")
    hex_values = re.findall(r"0x[0-9a-fA-F]{2}", m.group(1))
    data = bytes(int(v, 16) for v in hex_values)
    if len(data) != IMG_SIZE:
        raise RuntimeError(f"Expected {IMG_SIZE} bytes, got {len(data)}")
    return data

# ---------------------------------------------------------------------------
# 2. Write 8-bit grayscale BMP
# ---------------------------------------------------------------------------
def write_gray_bmp(path, pixel_data):
    """Write 128x128 8-bit grayscale BMP."""
    assert len(pixel_data) == IMG_SIZE

    row_size = IMG_WIDTH  # already 4-byte aligned
    pixel_array_size = row_size * IMG_HEIGHT
    palette_size = 256 * 4
    header_size = 14 + 40 + palette_size
    file_size = header_size + pixel_array_size

    # BMP File Header (14 bytes)
    file_header = struct.pack("<2sIHHI",
        b"BM",           # signature
        file_size,       # file size
        0,               # reserved
        0,               # reserved
        header_size      # offset to pixel data
    )

    # DIB Header (BITMAPINFOHEADER, 40 bytes)
    dib_header = struct.pack("<IiiHHIIiiII",
        40,              # header size
        IMG_WIDTH,       # width
        IMG_HEIGHT,      # height (positive = bottom-up)
        1,               # planes
        8,               # bits per pixel
        0,               # compression (BI_RGB)
        pixel_array_size,# image size
        2835,            # X pixels per meter (~72 DPI)
        2835,            # Y pixels per meter
        256,             # colors in palette
        0                # important colors
    )

    # Grayscale palette (256 * 4 bytes: B, G, R, 0)
    palette = b"".join(struct.pack("<BBBB", i, i, i, 0) for i in range(256))

    # Pixel data (bottom-up row order)
    # BMP rows are stored bottom-up; our data is top-down
    rows = [pixel_data[i * IMG_WIDTH:(i + 1) * IMG_WIDTH] for i in range(IMG_HEIGHT)]
    pixel_array = b"".join(reversed(rows))

    with open(path, "wb") as f:
        f.write(file_header + dib_header + palette + pixel_array)

# ---------------------------------------------------------------------------
# 3. PRNG-371 encryption via ctypes
# ---------------------------------------------------------------------------
def encrypt_image(pixel_data):
    """Encrypt pixel_data in-place using PRNG-371 (same seeds as sensor-node)."""
    lib = ctypes.CDLL(LIBPRNG)
    lib.prng371_init.argtypes = [ctypes.c_uint16] * 4
    lib.prng371_init.restype = None
    lib.prng371_crypt.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint16]
    lib.prng371_crypt.restype = None

    # Same seeds as sensor-node.c
    lib.prng371_init(1000, 2000, 3000, 4000)

    buf = ctypes.create_string_buffer(pixel_data, IMG_SIZE)
    lib.prng371_crypt(ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)), IMG_SIZE)
    return bytes(buf)[:IMG_SIZE]

# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(PAPER_FIGURES, exist_ok=True)

    # Parse original pixels
    original_data = parse_image_data()

    # Write original.bmp
    original_path = os.path.join(PAPER_FIGURES, "original.bmp")
    write_gray_bmp(original_path, original_data)
    print(f"[OK] {original_path}  ({IMG_WIDTH}x{IMG_HEIGHT} grayscale)")

    # Encrypt and write encrypted.bmp
    encrypted_data = encrypt_image(original_data)
    encrypted_path = os.path.join(PAPER_FIGURES, "encrypted.bmp")
    write_gray_bmp(encrypted_path, encrypted_data)
    print(f"[OK] {encrypted_path}  (PRNG-371 encrypted)")

    # Copy cooja_decrypted.png
    if os.path.exists(COOJA_DECRYPTED):
        dest = os.path.join(PAPER_FIGURES, "cooja_decrypted.png")
        shutil.copy2(COOJA_DECRYPTED, dest)
        print(f"[OK] {dest}  (Cooja end-to-end decryption result)")
    else:
        print(f"[WARN] {COOJA_DECRYPTED} not found, skipping")

    # Write README.md
    readme = os.path.join(PAPER_FIGURES, "README.md")
    with open(readme, "w") as f:
        f.write("""# Paper Figures — PRNG-371 Cooja End-to-End Simulation

This directory contains the image figures for the PRNG-371 lightweight encryption
end-to-end verification in Cooja.

## Current Image Set (new)

| File | Description |
|------|-------------|
| `original.bmp` | Original 128×128 grayscale image (`original new.bmp`, currently active). |
| `original new.bmp` | Same as `original.bmp` — the new active source image. |
| `encrypted.bmp` | PRNG-371 encrypted version of the new original image. Visually uniform noise. |
| `encrypted new.bmp` | Same as `encrypted.bmp` — encrypted version of the new image. |
| `cooja_decrypted.png` | Cooja decryption result for the new image (8 Relay nodes, 15% TX loss). |
| `cooja_decrypted new.png` | Same as `cooja_decrypted.png` — decrypted result for the new image. |

## Previous Image Set (woman / old)

| File | Description |
|------|-------------|
| `original_old.bmp` | Previous 128×128 grayscale image (`woman_128x128_gray8`). |
| `encrypted_old.bmp` | PRNG-371 encrypted version of the previous image. |
| `cooja_decrypted_old.png` | Previous Cooja decryption result. |

## Latest Metrics (new image)

- Total chunks: 256
- Received chunks: 223
- Lost chunks: 33
- Loss rate: 12.89%
- MSE: 3067.7997
- PSNR: 13.2625 dB

## Encryption Parameters

- Algorithm: PRNG-371 (6×16-bit state, 64-bit output)
- Seeds: `(x1, x2, x3, x4) = (1000, 2000, 3000, 4000)`
- Warmup: 2000 iterations
- Mode: XOR keystream (same key for encrypt/decrypt)

## Cooja Simulation Setup

- 10 nodes: 1 Sensor + 1 Gateway + 8 Relay
- Sensor→Gateway: UDGM `success_ratio_tx = 0.85` (15% packet loss)
- Gateway→Relay: multicast `ff02::1` with port isolation (9000–9007)
- Serial Socket: ports 60000 (Gateway) and 60001–60008 (Relays)
- Image: 16384 bytes = 256 chunks × 64 bytes

## Notes

- `original*.bmp` and `encrypted*.bmp` are generated using the **bit-exact** PRNG-371 C implementation (`libprng371.so`).
- `cooja_decrypted*.png` is produced from the actual Cooja headless simulation output and PC-side offline reconstruction.
- The new image set was generated after replacing `image_data.h` with `original new.bmp` and re-running the Cooja simulation.
""")
    print(f"[OK] {readme}")

if __name__ == "__main__":
    main()
