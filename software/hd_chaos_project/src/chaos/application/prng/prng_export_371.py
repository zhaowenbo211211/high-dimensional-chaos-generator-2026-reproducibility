#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
PRNG binary sequence export — A_odd=371, HLS bit-exact (via golden_prng).

  1. Dialog: 4 initial values + sequence length
  2. Create ChaosPRNG371 (C golden model, 100% bit-exact with HLS)
  3. Generate bits
  4. Save dialog (mode + path)
  5. Design report (LaTeX + enhancer + HW notes)

Run:
    python tests/prng_export_371.py
"""

import os
import tkinter as tk
from tkinter import messagebox, filedialog

from hw.src_371.golden_prng import ChaosPRNG371
from hw.src_371.golden_prng import _lib as _glib, GoldenState
import ctypes

# ------------------------------------------------------------------
# Parameter input dialog (initial state + length only, A_odd fixed 371)
# ------------------------------------------------------------------

def _ask_params(root):
    dialog = tk.Toplevel(root)
    dialog.title("PRNG Export — A_odd=371 (HLS Verified)")
    dialog.resizable(False, False)
    dialog.transient(root)
    dialog.grab_set()
    dialog.attributes('-topmost', True)

    result = {"x1": None, "x2": None, "x3": None, "x4": None, "length": None}

    # --- R1-LY params (fixed) ---
    info_frame = tk.LabelFrame(dialog, text="R1-LY Parameters (fixed)", padx=10, pady=8)
    info_frame.pack(padx=15, pady=(15, 5), fill="x")
    tk.Label(info_frame, text="A_odd=371, B_even=50, a_j1=[371,373,375], a_jn=[50,48,46], b=111",
             font=("", 9), fg="#006600").pack(anchor="w")

    # --- initial values ---
    init_frame = tk.LabelFrame(dialog, text="Initial State (0–65535, not all zero)",
                               padx=10, pady=8)
    init_frame.pack(padx=15, pady=(10, 5), fill="x")

    entries = {}
    defaults = {"x₁": 1000, "x₂": 2000, "x₃": 3000, "x₄": 4000}
    for col, (label, default) in enumerate(defaults.items()):
        tk.Label(init_frame, text=label, font=("", 10)).grid(row=1, column=col, padx=5)
        var = tk.StringVar(value=str(default))
        ent = tk.Entry(init_frame, textvariable=var, width=7, justify="center", font=("", 10))
        ent.grid(row=2, column=col, padx=5, pady=(2, 0))
        entries[label] = var

    # --- sequence length ---
    len_frame = tk.LabelFrame(dialog, text="Sequence Length", padx=10, pady=8)
    len_frame.pack(padx=15, pady=10, fill="x")
    tk.Label(len_frame, text="Bits to generate:", font=("", 10)).pack(side="left", padx=(0, 8))
    length_var = tk.StringVar(value="100000000")
    tk.Entry(len_frame, textvariable=length_var, width=14, justify="center",
             font=("", 10)).pack(side="left")
    tk.Label(len_frame, text="(default 10⁸)", font=("", 9), fg="gray").pack(side="left", padx=8)

    error_var = tk.StringVar()
    tk.Label(dialog, textvariable=error_var, fg="red", font=("", 9)).pack(pady=(0, 5))

    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=(5, 15))

    def on_ok():
        vals = []
        for label in ["x₁", "x₂", "x₃", "x₄"]:
            try:
                v = int(entries[label].get())
                if v < 0 or v > 65535: raise ValueError
                vals.append(v)
            except ValueError:
                error_var.set(f"{label} must be an integer in [0, 65535]")
                return
        if all(v == 0 for v in vals):
            error_var.set("Initial state cannot be all zeros")
            return
        try:
            lv = int(length_var.get())
            if lv < 64: raise ValueError
        except ValueError:
            error_var.set("Length must be at least 64 bits")
            return
        result["x1"], result["x2"], result["x3"], result["x4"] = vals
        result["length"] = lv
        dialog.destroy()

    tk.Button(btn_frame, text="OK", width=10, command=on_ok).pack(side="left", padx=8)
    tk.Button(btn_frame, text="Cancel", width=10, command=lambda: dialog.destroy()).pack(side="left", padx=8)

    dialog.geometry("420x340")
    dialog.update_idletasks()
    x = root.winfo_x() + (root.winfo_width() - 420) // 2
    y = root.winfo_y() + (root.winfo_height() - 340) // 2
    dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
    dialog.lift(); dialog.focus_force()
    dialog.wait_window()
    return result if result["x1"] is not None else None


# ------------------------------------------------------------------
# Mode selection dialog
# ------------------------------------------------------------------

def _ask_mode(root):
    dialog = tk.Toplevel(root)
    dialog.title("Choose Export Mode")
    dialog.resizable(False, False)
    dialog.transient(root); dialog.grab_set(); dialog.attributes('-topmost', True)
    result = [None]
    dialog.geometry("420x200")
    dialog.update_idletasks()
    x = root.winfo_x() + (root.winfo_width() - 420) // 2
    y = root.winfo_y() + (root.winfo_height() - 200) // 2
    dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
    dialog.lift(); dialog.focus_force()

    tk.Label(dialog, text="Choose binary sequence export format:", font=("", 11), pady=10).pack()
    f = tk.Frame(dialog); f.pack(pady=5)
    tk.Label(f, text="Raw binary:", font=("", 10, "bold")).grid(row=0, column=0, sticky="w", padx=10)
    tk.Label(f, text="8 bits/byte, compact (~12.5 MB)").grid(row=0, column=1, sticky="w")
    tk.Label(f, text="ASCII text:", font=("", 10, "bold")).grid(row=1, column=0, sticky="w", padx=10)
    tk.Label(f, text="'0'/'1' chars, readable (~100 MB)").grid(row=1, column=1, sticky="w")

    bf = tk.Frame(dialog); bf.pack(pady=15)
    def choose(m): result[0] = m; dialog.destroy()
    tk.Button(bf, text="Raw (.bin)", width=14, command=lambda: choose("raw")).pack(side="left", padx=8)
    tk.Button(bf, text="ASCII (.txt)", width=14, command=lambda: choose("ascii")).pack(side="left", padx=8)
    tk.Button(bf, text="Cancel", width=10, command=lambda: choose(None)).pack(side="left", padx=8)
    dialog.wait_window()
    return result[0]


# ------------------------------------------------------------------
# File writers
# ------------------------------------------------------------------

def _write_raw(filepath, words, length):
    total_bytes = (length + 7) // 8
    raw = bytearray(total_bytes)
    bit_idx = 0
    for w in words:
        w_int = int(w)
        for b in range(64):
            if bit_idx >= length: break
            raw[bit_idx // 8] |= ((w_int >> b) & 1) << (bit_idx % 8)
            bit_idx += 1
        if bit_idx >= length: break
    with open(filepath, "wb") as f: f.write(raw)

def _write_ascii(filepath, words, length):
    bit_idx = 0
    with open(filepath, "w") as f:
        for w in words:
            if bit_idx >= length: break
            remaining = min(64, length - bit_idx)
            chars = []
            w_int = int(w)
            for b in range(remaining):
                chars.append('1' if ((w_int >> b) & 1) else '0')
            f.write(''.join(chars))
            bit_idx += remaining
            if bit_idx % (10_000_000) < 64:
                print(f"    progress: {bit_idx:,}/{length:,} bits ({100.*bit_idx/length:.1f}%)")


# ------------------------------------------------------------------
# Design report (from prng_export.py, adapted for fixed A_odd=371)
# ------------------------------------------------------------------

def _print_design_report(x1, x2, x3, x4):
    print(f"\n{'='*70}")
    print(f"  CHAOS PRNG — Design Report (A_odd=371, HLS Verified)")
    print(f"{'='*70}")

    print(f"\n  [1] Chaotic System (R1-LY, A_odd=371, B_even=50)")
    print(f"  {'-'*60}")
    print(r"""  \[
  \left\{
  \begin{alignedat}{3}
  x_1(i+1) &= 371\,x_1(i) + 50\,x_4(i-2) \\
  x_2(i+1) &= 373\,x_2(i) + 48\,x_4(i-1) \\
  x_3(i+1) &= 375\,x_3(i) + 46\,x_4(i) \\
  x_4(i+1) &= 111\,x_2(i) + 4(x_3(i)+x_4(i))(1-(x_3(i)+x_4(i)))
  \end{alignedat}
  \right.
  \]""")
    print(f"  Δ = 99917.2 > 0 ✓  |  Lyapunov exponents: all > 0")

    print(f"\n  [2] Bit-Rotation Enhancer + Post-Processing")
    print(f"  {'-'*60}")
    print("    k_t = 1 + (t mod 15)")
    print("    Z = ROL64(x1∥x2∥x3∥x4, k_t)")
    print("    y_t = u_t ⊕ ROTL64(u_{t-1}, 17) ⊕ ROTL64(u_{t-2}, 43)")

    print(f"\n  [3] Hardware Implementation Notes")
    print(f"  {'-'*60}")
    print("  1. Constant multiplication → shift-and-add (zero DSP):")
    print("     371: (x<<8)+(x<<6)+(x<<5)+(x<<4)+(x<<1)+x")
    print("     373: (x<<8)+(x<<6)+(x<<5)+(x<<4)+(x<<2)+x")
    print("     375: (x<<8)+(x<<6)+(x<<5)+(x<<4)+(x<<2)+(x<<1)+x")
    print("      50: (x<<5)+(x<<4)+(x<<1)")
    print("      48: (x<<5)+(x<<4)")
    print("      46: (x<<5)+(x<<3)+(x<<2)+(x<<1)")
    print("     111: (x<<6)+(x<<5)+(x<<3)+(x<<2)+(x<<1)+x")
    print("  2. Logistic core: single DSP48E1 (variable × variable)")
    print("  3. Bit-rotation + ROTL: wiring only")
    print("  4. Total: 1 DSP / 1042 LUT / 707 FF / 0.13 W")
    print(f"  5. Init: [{x1},{x2},{x3},{x4}] | Warmup: 1000")
    print("  6. xc7z020clg400-1 @ 100 MHz, 800 MB/s (Vivado-verified)")
    print("  7. Software: golden_371.so (C shared lib, 100% HLS bit-exact)")
    print(f"{'='*70}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    root = tk.Tk()
    root.geometry("1x1+0+0")
    root.attributes('-topmost', True)
    root.update(); root.lift(); root.focus_force()

    params = _ask_params(root)
    if params is None:
        print("Cancelled.")
        root.destroy(); return

    x1, x2, x3, x4 = params["x1"], params["x2"], params["x3"], params["x4"]
    length = params["length"]

    print("=" * 60)
    print("ChaosPRNG-371 — Binary Sequence Export (HLS Bit-Exact)")
    print("=" * 60)
    print(f"  A_odd: 371 (fixed)  |  a_j1=[371,373,375]  |  b=111")
    print(f"  Initial state: [{x1}, {x2}, {x3}, {x4}]")
    print(f"  Sequence: {length:,} bits ({length/1e6:.1f}M)")

    # Create PRNG (C golden, HLS bit-exact)
    print(f"\n[1] Creating ChaosPRNG371 (C golden model, HLS bit-exact)...")
    prng = ChaosPRNG371(init_x=(x1, x2, x3, x4))

    # Generate
    print(f"\n[2] Generating {length:,} bits...")
    print("    Press Ctrl+C to abort.\n")
    words_needed = (length + 63) // 64
    print(f"    Words needed: {words_needed:,}")
    words = []
    try:
        for i in range(words_needed):
            words.append(prng.next())
            if (i + 1) % max(1, words_needed // 10) == 0:
                print(f"    progress: {i+1:,}/{words_needed:,} ({100.*(i+1)/words_needed:.1f}%)", flush=True)
    except KeyboardInterrupt:
        print("\n[ABORTED]"); root.destroy(); return
    print(f"\n    Done: {len(words):,} words")

    # Save dialog
    root.update(); root.lift(); root.focus_force()
    want = messagebox.askyesno("Save", f"Generated {length:,} bits. Save?", parent=root)
    if not want: print("\nCancelled."); root.destroy(); return

    mode = _ask_mode(root)
    if mode is None: root.destroy(); return

    root.update(); root.lift(); root.focus_force()
    ext = ".bin" if mode == "raw" else ".txt"
    fp = filedialog.asksaveasfilename(
        title="Save As", defaultextension=ext,
        filetypes=[("Binary files", "*.bin")] if mode == "raw" else [("Text files", "*.txt")],
        parent=root, initialfile=f"prng_371_{length}.{ext.lstrip('.')}")
    if not fp: print("\nCancelled."); root.destroy(); return
    root.destroy()

    print(f"\n[3] Writing {mode}...")
    if mode == "raw": _write_raw(fp, words, length)
    else: _write_ascii(fp, words, length)

    sz = os.path.getsize(fp)
    print(f"\n{'='*60}")
    print(f"Export complete!  File: {fp}  Size: {sz:,} bytes  Mode: {mode}")
    print(f"{'='*60}")

    _print_design_report(x1, x2, x3, x4)


if __name__ == "__main__":
    main()
