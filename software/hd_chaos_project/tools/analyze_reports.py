#!/usr/bin/env python3
# Maintainer: zhaowenbo
"""Chaos PRNG — Report Analyzer.

Parses HLS synthesis + Vivado implementation reports and prints a summary table.

Usage:
    python tools/analyze_reports.py [--hls-dir PATH] [--vivado-dir PATH]

Defaults scan the HLS project at ~/verilog-learning/chaos_prng/.
"""
import argparse, os, re, sys
from pathlib import Path
from typing import Optional, Dict


# ------------------------------------------------------------------
# Parsers
# ------------------------------------------------------------------

def parse_hls_report(path: str) -> Dict:
    """Parse Vitis HLS synthesis report."""
    if not os.path.exists(path):
        return {"error": f"not found: {path}"}

    text = Path(path).read_text()
    info = {}

    # HLS timing estimate
    m = re.search(r'ap_clk\s+\|\s+([\d.]+) ns\|\s+([\d.]+) ns', text)
    if m:
        info["hls_target_ns"] = float(m.group(1))
        info["hls_estim_ns"] = float(m.group(2))

    # Latency / Interval / Pipeline — format: | 4| 4| 74.512 ns| 74.512 ns| 5| 5| yes|
    block = text[text.find("Latency (cycles)"):] if "Latency (cycles)" in text else ""
    m = re.search(r'\|\s*(\d+)\|\s*(\d+)\|\s*[\d.]+\s*ns\|\s*[\d.]+\s*ns\|\s*(\d+)\|\s*(\d+)\|\s*(\w+)\|', block)
    if m:
        info["latency_min"] = int(m.group(1))
        info["latency_max"] = int(m.group(2))
        info["interval_min"] = int(m.group(3))
        info["interval_max"] = int(m.group(4))
        info["pipeline"] = m.group(5)

    # Utilization from HLS
    m = re.search(r'Total\s+\|\s+\d+\|\s+(\d+)\|\s+(\d+)\|\s+(\d+)', text)
    if m:
        info["hls_dsp"] = int(m.group(1))
        info["hls_ff"] = int(m.group(2))
        info["hls_lut"] = int(m.group(3))

    return info


def parse_vivado_util(path: str) -> Dict:
    """Parse Vivado utilization report."""
    if not os.path.exists(path):
        return {}
    text = Path(path).read_text()
    info = {}

    m = re.search(r'Slice LUTs\*?\s+\|\s+(\d+)\s*\|', text)
    if m:
        info["lut"] = int(m.group(1))
    m = re.search(r'Slice Registers\s+\|\s+(\d+)\s*\|', text)
    if m:
        info["ff"] = int(m.group(1))
    m = re.search(r'DSPs\s+\|\s+(\d+)\s*\|', text)
    if m:
        info["dsp"] = int(m.group(1))
    m = re.search(r'Block RAM Tile\s+\|\s+(\d+)\s*\|', text)
    if m:
        info["bram"] = int(m.group(1))

    # Design state
    m = re.search(r'Design State\s*:\s*(\w+)', text)
    if m:
        info["state"] = m.group(1)

    return info


def parse_vivado_timing(path: str) -> Dict:
    """Parse Vivado timing report."""
    if not os.path.exists(path):
        return {}
    text = Path(path).read_text()
    info = {}

    # WNS/TNS
    m = re.search(r'(?:ap_clk\s+)?([\d.-]+)\s+([\d.-]+)\s+(\d+)\s+(\d+)', text)
    # Try the intra-clock table first
    for line in text.split("\n"):
        if "ap_clk" in line and re.search(r'[\d.-]+\s+[\d.-]+\s+\d+\s+\d+', line):
            parts = line.split()
            # Find the WNS/TNS values in the line
            for i, p in enumerate(parts):
                if "ap_clk" in p and i + 4 < len(parts):
                    try:
                        info["wns"] = float(parts[i + 1])
                        info["tns"] = float(parts[i + 2])
                        info["failing"] = int(parts[i + 3])
                        info["total"] = int(parts[i + 4])
                        break
                    except (ValueError, IndexError):
                        continue

    # Clock period
    m = re.search(r'ap_clk\s+\{[\d.]+\s+([\d.]+)\}\s+([\d.]+)\s+([\d.]+)', text)
    if m:
        info["clock_period_ns"] = float(m.group(2))
        info["clock_freq_mhz"] = float(m.group(3))

    # Critical path logic levels
    m = re.search(r'Logic Levels:\s+(\d+)', text)
    if m:
        info["logic_levels"] = int(m.group(1))

    # Critical path data delay
    m = re.search(r'Data Path Delay:\s+([\d.]+)ns', text)
    if m:
        info["data_path_ns"] = float(m.group(1))

    # Design state
    m = re.search(r'Design State\s*:\s*(\w+)', text)
    if m:
        info["state"] = m.group(1)

    return info


def parse_vivado_power(path: str) -> Dict:
    """Parse Vivado power report."""
    if not os.path.exists(path):
        return {}
    text = Path(path).read_text()
    info = {}

    m = re.search(r'Total On-Chip Power \(W\)\s+\|\s+([\d.]+)', text)
    if m:
        info["total_w"] = float(m.group(1))
    m = re.search(r'Dynamic \(W\)\s+\|\s+([\d.]+)', text)
    if m:
        info["dynamic_w"] = float(m.group(1))
    m = re.search(r'Device Static \(W\)\s+\|\s+([\d.]+)', text)
    if m:
        info["static_w"] = float(m.group(1))
    m = re.search(r'Junction Temperature \(C\)\s+\|\s+([\d.]+)', text)
    if m:
        info["junction_c"] = float(m.group(1))
    m = re.search(r'Confidence Level\s+\|\s+(\w+)', text)
    if m:
        info["confidence"] = m.group(1)

    return info


# ------------------------------------------------------------------
# Display
# ------------------------------------------------------------------

HEADER = "\033[1;36m"  # cyan bold
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _pass_fail(val: float, threshold: float = 0.0) -> str:
    if val >= threshold:
        return f"{GREEN}PASS{RESET}"
    return f"{RED}FAIL{RESET}"


def _ok(val) -> str:
    return f"{GREEN}{val}{RESET}"


def _warn(val) -> str:
    return f"{YELLOW}{val}{RESET}"


def print_report(hls: Dict, util: Dict, timing: Dict, power: Dict):
    print()
    print(f"{HEADER}{'='*70}{RESET}")
    print(f"{HEADER}  Chaos PRNG — Report Analysis{RESET}")
    print(f"{HEADER}{'='*70}{RESET}")
    print()

    # ---- Performance ----
    print(f"{BOLD}Performance{RESET}")
    print(f"  {'Target clock:':30s} {hls.get('hls_target_ns', '?'):.1f} ns (HLS scheduling)  "
          f"{timing.get('clock_period_ns', '?'):.1f} ns (Vivado constraint)")
    print(f"  {'HLS timing estimate:':30s} {hls.get('hls_estim_ns', 0):.2f} ns")
    print(f"  {'Vivado data path:':30s} {timing.get('data_path_ns', 0):.2f} ns  "
          f"({timing.get('logic_levels', '?')} logic levels)")
    print(f"  {'Latency:':30s} {hls.get('latency_min','?')}–{hls.get('latency_max','?')} cycles")
    print(f"  {'Interval:':30s} {hls.get('interval_min','?')}–{hls.get('interval_max','?')}  "
          f"Pipeline: {hls.get('pipeline','?')}")
    print()

    # ---- Timing ----
    print(f"{BOLD}Timing @ {timing.get('clock_period_ns', '?'):.1f}ns ({timing.get('clock_freq_mhz', 0):.0f} MHz){RESET}")
    wns = timing.get("wns", -999)
    print(f"  {'Setup WNS:':30s} {wns:+.3f} ns  {_pass_fail(wns)}  "
          f"({timing.get('failing','?')}/{timing.get('total','?')} endpoints)")
    print(f"  {'TNS:':30s} {timing.get('tns', 0):.3f} ns")
    print(f"  {'Hold WNS:':30s} (see report — routing fixes hold)")
    print()

    # ---- Resource ----
    print(f"{BOLD}Resource Utilization (xc7z020clg400-1){RESET}")
    print(f"  {'Resource':12s} {'Vivado':>8s} {'HLS Est':>8s} {'Available':>10s} {'Util%':>7s}")
    print(f"  {'-'*47}")
    for name, v_key, h_key, avail in [
        ("LUT", "lut", "hls_lut", 53200),
        ("FF", "ff", "hls_ff", 106400),
        ("DSP", "dsp", "hls_dsp", 220),
        ("BRAM", "bram", "hls_bram", 280),
    ]:
        v = util.get(v_key, "?")
        h = hls.get(h_key, "?")
        pct = f"{int(v)/avail*100:.1f}%" if isinstance(v, int) and v != "?" else "?"
        print(f"  {name:12s} {str(v):>8s} {str(h):>8s} {avail:>10,d} {pct:>7s}")
    print()

    # ---- Power ----
    print(f"{BOLD}Power (Vivado estimate, {power.get('state','?')}){RESET}")
    print(f"  {'Total:':12s} {power.get('total_w', 0):.3f} W  "
          f"(dynamic {power.get('dynamic_w',0):.3f} + static {power.get('static_w',0):.3f})")
    print(f"  {'Junction temp:':12s} {power.get('junction_c','?')} °C  "
          f"Confidence: {power.get('confidence','?')}")
    print()

    # ---- Summary verdict ----
    print(f"{HEADER}{'='*70}{RESET}")
    if wns >= 0:
        print(f"  {GREEN}VERDICT: Timing MET at {timing.get('clock_period_ns','?')}ns — "
              f"ready for place & route{RESET}")
    elif wns > -0.5:
        print(f"  {YELLOW}VERDICT: Timing ALMOST met (WNS={wns:+.3f}ns). "
              f"Increase clock period by {abs(wns)+0.2:.1f}ns.{RESET}")
    else:
        print(f"  {RED}VERDICT: Timing FAILED (WNS={wns:+.3f}ns). "
              f"Need {timing.get('data_path_ns',1)-wns:.1f}ns clock.{RESET}")
    print(f"{HEADER}{'='*70}{RESET}")
    print()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chaos PRNG report analyzer")
    parser.add_argument("--hls-dir", default=os.path.expanduser(
        "~/verilog-learning/chaos_prng"),
        help="HLS project root (default: ~/verilog-learning/chaos_prng)")
    parser.add_argument("--vivado-dir", default=None,
        help="Vivado output dir (default: HLS_DIR/output/implementation)")
    args = parser.parse_args()

    hls_root = Path(args.hls_dir)
    synth_rpt = hls_root / "output" / "synthesis" / "report.txt"
    vivado_dir = Path(args.vivado_dir) if args.vivado_dir \
        else hls_root / "output" / "implementation"

    print(f"  HLS report:    {synth_rpt}")
    print(f"  Vivado reports: {vivado_dir}")

    hls = parse_hls_report(str(synth_rpt))
    util = parse_vivado_util(str(vivado_dir / "utilization_impl.rpt"))
    timing = parse_vivado_timing(str(vivado_dir / "timing_impl.rpt"))
    power = parse_vivado_power(str(vivado_dir / "power_impl.rpt"))

    print_report(hls, util, timing, power)


if __name__ == "__main__":
    main()
