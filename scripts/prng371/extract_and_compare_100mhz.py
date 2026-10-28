#!/usr/bin/env python3
# Maintainer: zhaowenbo
"""Extract PRNG sequence from Vivado ILA CSV aligned to cfg_done rising edge,
deduplicate consecutive samples, skip warmup, and compare with golden reference."""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def parse_ila_csv(csv_path):
    """Parse Vivado ILA CSV with radix row. Return list of dicts."""
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        _radix = next(reader)
        for row in reader:
            if not row:
                continue
            rows.append(dict(zip(header, row)))
    return rows


def extract_prng_sequence(rows, warmup=2000):
    """Find cfg_done rising edge, extract unique PRNG outputs, skip warmup."""
    # Find first row where cfg_done == 1
    start_idx = None
    prev_cfg_done = '0'
    for i, row in enumerate(rows):
        cfg_done = row.get('cfg_done', '0').strip().lower()
        if prev_cfg_done == '0' and cfg_done == '1':
            start_idx = i
            break
        prev_cfg_done = cfg_done

    if start_idx is None:
        raise RuntimeError("cfg_done rising edge not found in capture")

    print(f"cfg_done rising edge at buffer sample {rows[start_idx]['Sample in Buffer']}")

    # Extract ila_probe2_diag values from cfg_done high onwards
    # Deduplicate consecutive identical values (ILA may sample at 2x PRNG rate)
    values = []
    last_val = None
    for row in rows[start_idx:]:
        val_str = row.get('ila_probe2_diag[63:0]', '').strip()
        if not val_str:
            continue
        try:
            val = int(val_str, 16)
        except ValueError:
            continue
        if val != last_val:
            values.append(val)
            last_val = val

    # Skip the leading zero that appears immediately after cfg_done goes high
    # (one-cycle latency before first valid PRNG output)
    while values and values[0] == 0:
        values = values[1:]

    print(f"Total unique PRNG outputs after cfg_done: {len(values)}")
    if len(values) < warmup + 100:
        raise RuntimeError(f"Not enough outputs: {len(values)} < {warmup} + 100")

    # Skip warmup
    post_warmup = values[warmup:]
    print(f"Post-warmup outputs: {len(post_warmup)}")
    return post_warmup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True, help='ILA CSV file')
    parser.add_argument('--golden', required=True, help='Golden reference txt file')
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--output-txt', default='fpga_post_warmup.txt')
    parser.add_argument('--compare-script', default='prng_experiment/tools/compare_fpga_capture.py')
    parser.add_argument('--output-report', default='REAL_FPGA_COMPARE_REPORT.md')
    args = parser.parse_args()

    rows = parse_ila_csv(args.capture)
    post_warmup = extract_prng_sequence(rows, warmup=args.warmup)

    # Write post-warmup sequence to txt
    with open(args.output_txt, 'w') as f:
        for val in post_warmup:
            f.write(f"{val:016x}\n")

    print(f"Wrote {len(post_warmup)} post-warmup words to {args.output_txt}")

    # Run compare_fpga_capture.py
    cmd = [
        sys.executable,
        args.compare_script,
        '--golden', args.golden,
        '--capture', args.output_txt,
        '--format', 'txt',
        '--output_report', args.output_report,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
