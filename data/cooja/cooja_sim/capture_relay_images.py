#!/usr/bin/env python3
# Maintainer: zhaowenbo
"""
Capture serial output from Gateway (port 60000) and 8 Relay nodes (ports 60001-60008).

Usage:
    python3 capture_relay_images.py

- Gateway output is saved to raw_logs/gateway_serial.log and parsed for bitmap.
- Relay hex output is merged into a single 128x128 grayscale PNG.

Note: This script keeps sockets open until the remote side closes them (or
until killed externally). This avoids triggering a Cooja SerialSocketServer
cleanup bug that can stop the simulation when clients disconnect early.
"""

import os
import re
import socket
import threading
import time
import json

import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE, "..")
RAW_LOGS = os.path.join(PROJECT_ROOT, "raw_logs")

GATEWAY_PORT = 60000
RELAY_PORTS = [60001 + i for i in range(8)]
PART_SIZE = 2048
TIMEOUT = 180

os.makedirs(RAW_LOGS, exist_ok=True)


def connect_port(port, label):
    """Connect to a Serial Socket port, retrying until success."""
    sock = None
    retries = 0
    while sock is None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(('127.0.0.1', port))
            print(f"{label} port {port}: connected")
            return sock
        except (ConnectionRefusedError, OSError):
            if sock:
                sock.close()
                sock = None
            retries += 1
            if retries % 10 == 1:
                print(f"{label} port {port}: waiting for Cooja ({retries})...")
            time.sleep(0.5)
        except Exception as e:
            print(f"{label} port {port}: unexpected error: {e}")
            return None


def receive_gateway():
    """Connect to Gateway Serial Socket (port 60000) and capture all output."""
    sock = connect_port(GATEWAY_PORT, "Gateway")
    if sock is None:
        return b""

    sock.settimeout(TIMEOUT)
    buffer = b""
    end_seen = False
    try:
        while True:
            data = sock.recv(4096)
            if not data:
                break
            buffer += data
            # Keep reading after distribution complete; let Cooja close the socket.
            if (not end_seen and
                (b"=== DISTRIBUTION COMPLETE ===" in buffer or b"=== SIMULATION COMPLETE ===" in buffer)):
                end_seen = True
                print(f"Gateway {GATEWAY_PORT}: end marker seen, keeping socket open...")
    except socket.timeout:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass

    # Save raw gateway serial log
    with open(os.path.join(RAW_LOGS, "gateway_serial.log"), "wb") as f:
        f.write(buffer)

    print(f"Gateway {GATEWAY_PORT}: captured {len(buffer)} bytes")
    return buffer


def parse_gateway_bitmap(buffer):
    """Parse bitmap string from Gateway serial output."""
    text = buffer.decode('utf-8', errors='ignore')
    # Look for bitmap line: "bitmap = 1110111..."
    m = re.search(r'bitmap\s*=\s*([01]{256})', text)
    if m:
        bitmap_str = m.group(1)
        received = [i for i, c in enumerate(bitmap_str) if c == '1']
        lost = [i for i, c in enumerate(bitmap_str) if c == '0']
        return bitmap_str, received, lost

    # Fallback: look for individual "received chunk X" lines
    received = sorted(set(int(m.group(1)) for m in re.finditer(r'received chunk (\d+)', text)))
    lost = [i for i in range(256) if i not in received]
    bitmap_str = "".join("1" if i in received else "0" for i in range(256))
    return bitmap_str, received, lost


def receive_relay(port, idx):
    """Connect to one relay's Serial Socket and capture its hex output."""
    sock = connect_port(port, f"Relay-{idx}")
    if sock is None:
        return b""

    sock.settimeout(TIMEOUT)
    buffer = ""
    end_seen = False
    try:
        while True:
            data = sock.recv(4096)
            if not data:
                break
            buffer += data.decode('utf-8', errors='ignore')
            if not end_seen and ">>>END_PART<<<" in buffer:
                end_seen = True
                print(f"Relay port {port}: END_PART seen, keeping socket open...")
    except socket.timeout:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass

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

    nonzero = len(bytes(part).rstrip(b'\x00'))
    print(f"Relay port {port}: received {nonzero} non-zero bytes")
    return bytes(part)


def main():
    # Start all connections concurrently
    results = {}
    threads = []

    def gateway_worker():
        results['gateway'] = receive_gateway()

    def relay_worker(idx):
        results[f'relay_{idx}'] = receive_relay(RELAY_PORTS[idx], idx)

    print("Connecting to all Serial Sockets (Gateway + 8 Relays)...")
    t = threading.Thread(target=gateway_worker)
    threads.append(t)
    t.start()

    for i in range(8):
        t = threading.Thread(target=relay_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Parse Gateway bitmap
    gateway_buffer = results.get('gateway', b'')
    bitmap_str, received, lost = parse_gateway_bitmap(gateway_buffer)

    print(f"\nGateway bitmap parsed:")
    print(f"  Total chunks: 256")
    print(f"  Received: {len(received)}")
    print(f"  Lost: {len(lost)}")
    if lost:
        print(f"  Lost indices: {lost}")

    # Save parsed bitmap info
    with open(os.path.join(RAW_LOGS, "gateway_bitmap.json"), "w") as f:
        json.dump({
            "bitmap": bitmap_str,
            "received_chunks": received,
            "lost_chunks": lost,
            "total_chunks": 256,
            "received_count": len(received),
            "lost_count": len(lost),
            "loss_rate": round(len(lost) / 256, 6),
        }, f, indent=2)

    # Merge relay parts
    parts = [results.get(f'relay_{i}', b'') for i in range(8)]
    image_data = b"".join(parts)
    print(f"\nTotal merged: {len(image_data)} bytes (expected {128*128})")

    if len(image_data) < 128 * 128:
        print("WARNING: missing data, padding with zeros")
        image_data = image_data + b'\x00' * (128*128 - len(image_data))

    image = np.frombuffer(image_data[:128*128], dtype=np.uint8).reshape((128, 128))
    img = Image.fromarray(image, mode='L')
    img.save('cooja_decrypted.png')
    print(f"Saved cooja_decrypted.png ({img.size[0]}x{img.size[1]} grayscale)")

    # Keep main thread alive briefly to allow final socket flushes, then exit.
    # If running under run_experiment.sh, this process will be terminated after
    # Cooja finishes. Keeping sockets open until then avoids the Cooja cleanup bug.
    print("Main thread done. Keeping process alive until external termination...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
