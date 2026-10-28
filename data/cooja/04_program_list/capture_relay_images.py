#!/usr/bin/env python3
# Maintainer: zhaowenbo
"""
Capture image parts from 8 relay nodes via Cooja Serial Socket.

Usage:
    python3 capture_relay_images.py

Connects to ports 60001~60008, receives hex output from each relay node,
merges 8 parts into a single 128x128 grayscale PNG.
"""

import re
import socket
import threading
import time
import numpy as np
from PIL import Image

PORTS = [60001 + i for i in range(8)]
PART_SIZE = 2048
TIMEOUT = 60


def receive_part(port):
    """Connect to one relay's Serial Socket and capture its hex output.
    Retries until Cooja starts listening."""
    sock = None
    retries = 0
    while sock is None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(('127.0.0.1', port))
            print(f"Port {port}: connected")
        except (ConnectionRefusedError, OSError) as e:
            if sock:
                sock.close()
                sock = None
            retries += 1
            if retries % 5 == 1:
                print(f"Port {port}: waiting for Cooja ({retries})...")
            time.sleep(1.0)
        except Exception as e:
            print(f"Port {port}: unexpected error: {e}")
            return b''

    # Once connected, use longer timeout for receiving data
    sock.settimeout(TIMEOUT)
    buffer = ""
    try:
        while True:
            data = sock.recv(4096)
            if not data:
                break
            buffer += data.decode('utf-8', errors='ignore')
            if ">>>END_PART<<<" in buffer:
                break
    except socket.timeout:
        pass
    finally:
        sock.close()

    # Parse hex lines: R000:7f3a2b...
    pattern = re.compile(r'R(\d{3}):([0-9a-fA-F]+)')
    part = bytearray(PART_SIZE)
    for match in pattern.finditer(buffer):
        line_idx = int(match.group(1))
        hex_data = match.group(2)
        data = bytes.fromhex(hex_data)
        offset = line_idx * 48
        for i, byte in enumerate(data):
            if offset + i < PART_SIZE:
                part[offset + i] = byte

    print(f"Port {port}: received {len(bytes(part).rstrip(b'\x00'))} non-zero bytes")
    return bytes(part)


def main():
    parts = [b''] * 8
    threads = []

    def worker(idx):
        parts[idx] = receive_part(PORTS[idx])
        print(f"Port {PORTS[idx]}: received {len(parts[idx])} bytes")

    print("Connecting to 8 relay Serial Sockets...")
    for i in range(8):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Merge 8 parts
    image_data = b''.join(parts)
    print(f"Total merged: {len(image_data)} bytes (expected {128*128})")

    if len(image_data) < 128 * 128:
        print("WARNING: missing data, only " + str(len(image_data)) + " bytes received")
        image_data = image_data + b'\x00' * (128*128 - len(image_data))

    image = np.frombuffer(image_data[:128*128], dtype=np.uint8).reshape((128, 128))
    img = Image.fromarray(image, mode='L')
    img.save('cooja_decrypted.png')
    print(f"Saved cooja_decrypted.png ({img.size[0]}x{img.size[1]} grayscale)")


if __name__ == '__main__':
    main()
