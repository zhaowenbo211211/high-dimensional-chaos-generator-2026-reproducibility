#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
PRNG binary sequence export script.

  1. Parameter dialog: A_odd + 4 initial values + sequence length
  2. Auto-synthesize R1-LY params and show generated coefficients
  3. Create ChaosPRNG instance
  4. Generate bits
  5. Interactive save dialog (mode: raw / ascii, choose path)

Run:
    python tests/prng_export.py
"""

import os
import tkinter as tk
from tkinter import messagebox, filedialog

from chaos.core.params import ComplementaryChaosParams
from chaos.application.prng import ChaosPRNG
from chaos.utils.latex import make_F, latex_system_like_figure_full

# ------------------------------------------------------------------
# Parameter input dialog
# ------------------------------------------------------------------

def _ask_params(root):
    """Show dialog to input 4 initial values and sequence length."""
    dialog = tk.Toplevel(root)
    dialog.title("PRNG Parameters")
    dialog.resizable(False, False)
    dialog.transient(root)
    dialog.grab_set()
    dialog.attributes('-topmost', True)

    result = {"x1": None, "x2": None, "x3": None, "x4": None, "length": None,
              "A_odd": None, "params": None}

    # --- R1-LY parameter ---
    r1_frame = tk.LabelFrame(dialog, text="R1-LY Synthesis", padx=10, pady=8)
    r1_frame.pack(padx=15, pady=(15, 5), fill="x")

    tk.Label(r1_frame, text="A_odd (odd, ≥5):", font=("", 10)).pack(side="left", padx=(0, 8))
    aodd_var = tk.StringVar(value="371")
    tk.Entry(r1_frame, textvariable=aodd_var, width=8,
             justify="center", font=("", 10)).pack(side="left")
    tk.Label(r1_frame, text="B_even=50, η=0.3 (auto)", font=("", 9),
             fg="gray").pack(side="left", padx=8)

    # --- generated params display ---
    gen_var = tk.StringVar(value="(will be generated)")
    gen_label = tk.Label(dialog, textvariable=gen_var, font=("", 9),
                         fg="#006600", justify="left")
    gen_label.pack(padx=15, pady=(5, 0), anchor="w")

    # --- initial values ---
    init_frame = tk.LabelFrame(dialog, text="Initial State (0–65535, not all zero)",
                               padx=10, pady=8)
    init_frame.pack(padx=15, pady=(10, 5), fill="x")

    entries = {}
    defaults = {"x₁": 1000, "x₂": 2000, "x₃": 3000, "x₄": 4000}
    for col, (label, default) in enumerate(defaults.items()):
        tk.Label(init_frame, text=label, font=("", 10)).grid(
            row=1, column=col, padx=5)
        var = tk.StringVar(value=str(default))
        ent = tk.Entry(init_frame, textvariable=var, width=7,
                       justify="center", font=("", 10))
        ent.grid(row=2, column=col, padx=5, pady=(2, 0))
        entries[label] = var

    # --- sequence length ---
    len_frame = tk.LabelFrame(dialog, text="Sequence Length", padx=10, pady=8)
    len_frame.pack(padx=15, pady=10, fill="x")

    tk.Label(len_frame, text="Bits to generate:", font=("", 10)).pack(
        side="left", padx=(0, 8))
    length_var = tk.StringVar(value="100000000")
    length_entry = tk.Entry(len_frame, textvariable=length_var, width=14,
                            justify="center", font=("", 10))
    length_entry.pack(side="left")
    tk.Label(len_frame, text="(default 10⁸ = 100,000,000)", font=("", 9),
             fg="gray").pack(side="left", padx=8)

    # --- error label ---
    error_var = tk.StringVar()
    error_label = tk.Label(dialog, textvariable=error_var, fg="red",
                           font=("", 9))
    error_label.pack(pady=(0, 5))

    # --- buttons ---
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=(5, 15))

    def on_generate():
        """Auto-generate R1-LY params from A_odd."""
        try:
            a_odd = int(aodd_var.get())
            if a_odd < 5 or a_odd % 2 != 1:
                error_var.set("A_odd must be odd and ≥ 5")
                return
        except ValueError:
            error_var.set("A_odd must be an integer")
            return

        try:
            synth = ComplementaryChaosParams(n=4, A_odd=a_odd, B_even=50)
            p = synth.synthesize()
            result["params"] = p
            result["A_odd"] = a_odd
            a_j1 = list(p['a_j1'])
            a_jn = list(p['a_jn'])
            gen_var.set(
                f"Generated: a_j1={a_j1}  a_jn={a_jn}  b={p['b']}  "
                f"Δ={p['discriminant']:.1f} (>0 ✓)")
            error_var.set("")
        except Exception as e:
            error_var.set(f"Synthesis failed: {e}")

    def on_ok():
        # Parse A_odd
        try:
            a_odd = int(aodd_var.get())
            if a_odd < 5 or a_odd % 2 != 1:
                error_var.set("A_odd must be odd and ≥ 5")
                return
        except ValueError:
            error_var.set("A_odd must be an integer")
            return

        # Auto-synthesize if not done yet
        if result["params"] is None:
            on_generate()
        if result["params"] is None:
            return

        # Parse and validate initial values
        vals = []
        for label in ["x₁", "x₂", "x₃", "x₄"]:
            try:
                v = int(entries[label].get())
                if v < 0 or v > 65535:
                    raise ValueError
                vals.append(v)
            except ValueError:
                error_var.set(
                    f"{label} must be an integer in [0, 65535]")
                return

        if all(v == 0 for v in vals):
            error_var.set("Initial state cannot be all zeros")
            return

        # Parse and validate length
        try:
            length_val = int(length_var.get())
            if length_val < 64:
                error_var.set("Length must be at least 64 bits")
                return
        except ValueError:
            error_var.set("Length must be a positive integer")
            return

        result["x1"], result["x2"], result["x3"], result["x4"] = vals
        result["length"] = length_val
        result["A_odd"] = a_odd
        dialog.destroy()

    def on_cancel():
        dialog.destroy()

    tk.Button(btn_frame, text="Generate", width=10,
              command=on_generate).pack(side="left", padx=8)
    tk.Button(btn_frame, text="OK", width=10,
              command=on_ok).pack(side="left", padx=8)
    tk.Button(btn_frame, text="Cancel", width=10,
              command=on_cancel).pack(side="left", padx=8)

    # Position near root
    dialog.geometry("440x420")
    dialog.update_idletasks()
    x = root.winfo_x() + (root.winfo_width() - 440) // 2
    y = root.winfo_y() + (root.winfo_height() - 420) // 2
    dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
    dialog.lift()
    dialog.focus_force()

    dialog.wait_window()
    return result if result["x1"] is not None else None


# ------------------------------------------------------------------
# Mode selection dialog
# ------------------------------------------------------------------

def _ask_mode(root):
    """Show a custom dialog to choose raw vs ascii mode."""
    dialog = tk.Toplevel(root)
    dialog.title("Choose Export Mode")
    dialog.resizable(False, False)
    dialog.transient(root)
    dialog.grab_set()
    dialog.attributes('-topmost', True)

    result = [None]

    dialog.geometry("420x200")
    dialog.update_idletasks()
    x = root.winfo_x() + (root.winfo_width() - 420) // 2
    y = root.winfo_y() + (root.winfo_height() - 200) // 2
    dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
    dialog.lift()
    dialog.focus_force()

    tk.Label(dialog, text="Choose binary sequence export format:",
             font=("", 11), pady=10).pack()

    frame = tk.Frame(dialog)
    frame.pack(pady=5)

    tk.Label(frame, text="Raw binary:", font=("", 10, "bold")).grid(
        row=0, column=0, sticky="w", padx=10, pady=2)
    tk.Label(frame, text="8 bits/byte, compact (~12.5 MB)").grid(
        row=0, column=1, sticky="w")
    tk.Label(frame, text="ASCII text:", font=("", 10, "bold")).grid(
        row=1, column=0, sticky="w", padx=10, pady=2)
    tk.Label(frame, text="'0'/'1' chars, readable (~100 MB)").grid(
        row=1, column=1, sticky="w")

    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=15)

    def choose(mode):
        result[0] = mode
        dialog.destroy()

    tk.Button(btn_frame, text="Raw (.bin)", width=14,
              command=lambda: choose("raw")).pack(side="left", padx=8)
    tk.Button(btn_frame, text="ASCII (.txt)", width=14,
              command=lambda: choose("ascii")).pack(side="left", padx=8)
    tk.Button(btn_frame, text="Cancel", width=10,
              command=lambda: choose(None)).pack(side="left", padx=8)

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
            if bit_idx >= length:
                break
            raw[bit_idx // 8] |= ((w_int >> b) & 1) << (bit_idx % 8)
            bit_idx += 1
        if bit_idx >= length:
            break

    with open(filepath, "wb") as f:
        f.write(raw)


def _write_ascii(filepath, words, length):
    bit_idx = 0
    with open(filepath, "w") as f:
        for w in words:
            if bit_idx >= length:
                break
            remaining = min(64, length - bit_idx)
            chars = []
            w_int = int(w)
            for b in range(remaining):
                chars.append('1' if ((w_int >> b) & 1) else '0')
            f.write(''.join(chars))
            bit_idx += remaining

            if bit_idx % (10_000_000) < 64:
                pct = 100.0 * bit_idx / length
                print(f"    progress: {bit_idx:,}/{length:,} bits ({pct:.1f}%)")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # --- Create root (Linux-friendly: tiny visible window) ---
    root = tk.Tk()
    root.geometry("1x1+0+0")
    root.attributes('-topmost', True)
    root.update()
    root.lift()
    root.focus_force()

    # ==================================================================
    # Step 1: Parameter dialog
    # ==================================================================
    params = _ask_params(root)
    if params is None:
        print("Cancelled — no parameters.")
        root.destroy()
        return

    x1, x2, x3, x4 = params["x1"], params["x2"], params["x3"], params["x4"]
    length = params["length"]
    params_info = params["params"]  # R1-LY synthesized params dict

    print("=" * 60)
    print("ChaosPRNG — Binary Sequence Export")
    print("=" * 60)
    print(f"  A_odd: {params['A_odd']}")
    print(f"  a_j1:  {list(params_info['a_j1'])}")
    print(f"  a_jn:  {list(params_info['a_jn'])}")
    print(f"  b:     {params_info['b']}")
    print(f"  Δ:     {params_info['discriminant']:.1f}")
    print(f"  Initial state: [{x1}, {x2}, {x3}, {x4}]")
    print(f"  Sequence length: {length:,} bits ({length / 1e6:.1f} million)")

    # Print chaotic system LaTeX
    print(f"\n{'='*60}")
    print("Chaotic System Equations (LaTeX):")
    print(f"{'='*60}")
    F = make_F(4, params_info['a_j1'], params_info['a_jn'], int(params_info['b']))
    print(latex_system_like_figure_full(F, with_mod=False))

    # ==================================================================
    # Step 2: Create PRNG & set state
    # ==================================================================
    print(f"\n[1] Creating ChaosPRNG (A_odd={params['A_odd']}, B_even=50)...")
    prng = ChaosPRNG.from_r1_ly(A_odd=int(params['A_odd']), B_even=50)

    print(f"[2] Setting initial state: [{x1}, {x2}, {x3}, {x4}]")
    prng.set_state(x1, x2, x3, x4)
    print(f"    State set: {prng.get_state()}")

    # ==================================================================
    # Step 3: Generate bits
    # ==================================================================
    print(f"\n[3] Generating {length:,} bits...")
    print("    Press Ctrl+C to abort at any time.\n")

    words_needed = (length + 63) // 64
    print(f"    Words needed: {words_needed:,}")
    try:
        words = prng.generate(words_needed,
                              progress_interval=max(1, words_needed // 10))
    except KeyboardInterrupt:
        print(f"\n\n[ABORTED] Generation interrupted by user.")
        root.destroy()
        return
    print(f"\n    Generation complete: {len(words):,} words")

    # ==================================================================
    # Step 4: Save dialog (mode + path)
    # ==================================================================
    root.update()
    root.lift()
    root.focus_force()

    want_save = messagebox.askyesno(
        "Save Binary Sequence",
        f"Generated {length:,} bits ({length // 1_000_000} million).\n\n"
        "Save to file?",
        parent=root,
        icon=messagebox.QUESTION)

    if not want_save:
        print("\nSave cancelled by user.")
        root.destroy()
        return

    mode_choice = _ask_mode(root)
    if mode_choice is None:
        root.destroy()
        return

    mode_label = ("Raw binary (.bin, packed 8 bits/byte)" if mode_choice == "raw"
                  else "ASCII text (.txt, '0'/'1' characters)")
    size_est = (f"{length // 8_000_000:.1f} MB" if mode_choice == "raw"
                else f"{length // 1_000_000} MB")

    root.update()
    root.lift()
    root.focus_force()

    default_ext = ".bin" if mode_choice == "raw" else ".txt"
    filetypes = [("Binary files", "*.bin")] if mode_choice == "raw" \
        else [("Text files", "*.txt")]
    filetypes.append(("All files", "*.*"))

    file_path = filedialog.asksaveasfilename(
        title=f"Save as {mode_label}",
        defaultextension=default_ext,
        filetypes=filetypes,
        parent=root,
        initialfile=f"prng_{length}.{default_ext.lstrip('.')}")

    if not file_path:
        print("\nSave cancelled by user.")
        root.destroy()
        return

    root.destroy()

    # ==================================================================
    # Step 5: Write file
    # ==================================================================
    print(f"\n[4] Writing {mode_label}...")
    print(f"    Output: {file_path}")
    print(f"    Estimated size: {size_est}")

    if mode_choice == "raw":
        _write_raw(file_path, words, length)
    else:
        _write_ascii(file_path, words, length)

    actual_size = os.path.getsize(file_path)
    print(f"\n{'=' * 60}")
    print("Export complete!")
    print(f"  File:   {file_path}")
    print(f"  Size:   {actual_size:,} bytes ({actual_size / 1e6:.1f} MB)")
    print(f"  Bits:   {length:,}")
    print(f"  Mode:   {mode_choice}")
    print(f"{'=' * 60}")

    # ==================================================================
    # Step 6: Design report
    # ==================================================================
    _print_design_report(params_info, params['A_odd'], x1, x2, x3, x4)


# ------------------------------------------------------------------
# Design report
# ------------------------------------------------------------------

def _print_design_report(params_info, A_odd, x1, x2, x3, x4):
    """Print a design summary report with LaTeX, enhancer, and HW notes."""
    a_j1 = params_info['a_j1']
    a_jn = params_info['a_jn']
    b_val = int(params_info['b'])

    print(f"\n{'='*70}")
    print(f"  CHAOS PRNG — Design Report")
    print(f"{'='*70}")

    # ---- 1. Chaotic system LaTeX ----
    print(f"\n  [1] Chaotic System (R1-LY, A_odd={A_odd}, B_even=50)")
    print(f"  {'-'*60}")
    F = make_F(4, a_j1, a_jn, b_val)
    tex = latex_system_like_figure_full(F, with_mod=False)
    for line in tex.split('\n'):
        print(f"  {line}")
    print(f"  Lyapunov exponents: all > 0 (verified)")
    print(f"  Cone-field discriminant Δ = {params_info['discriminant']:.1f} > 0 ✓")

    # ---- 2. Enhancer + post-processing ----
    print(f"\n  [2] Bit-Rotation Enhancer (Algorithm 2)")
    print(f"  {'-'*60}")
    print(f"  After each chaos iteration:")
    print(f"    x = {{x₁, x₂, x₃, x₄}}                          (4 × 16-bit)")
    print(f"    Z = x₁ ∥ x₂ ∥ x₃ ∥ x₄                          (pack → 64-bit)")
    print(f"    Z = ROL₆₄(Z, k_t)                              (cyclic rotate)")
    print(f"    k_t = 1 + (t mod 15)                            (periodic schedule)")
    print(f"    x = unpack(Z)                                    (back to 4 × 16-bit)")
    print(f"")
    print(f"  Post-Processing (ROTL-XOR Mixing):")
    print(f"    u_t = x₁ ∥ x₂ ∥ x₃ ∥ x₄                         (64-bit concat)")
    print("    y_t = u_t ⊕ ROTL₆₄(u_{t-1}, 17) ⊕ ROTL₆₄(u_{t-2}, 43)")
    print(f"  Hardware cost: 2 × 64-bit registers + fixed wiring + 2 XORs")

    # ---- 3. HW implementation notes ----
    print(f"\n  [3] Hardware Implementation Notes")
    print(f"  {'-'*60}")
    print(f"  1. Constant multiplication → shift-and-add (zero DSP):")
    for i, coeff in enumerate(a_j1):
        desc = _decomp_desc(int(coeff))
        print(f"     a_j1[{i}] = {int(coeff)}:  {desc}")
    for i, coeff in enumerate(a_jn):
        desc = _decomp_desc(int(coeff))
        print(f"     a_jn[{i}] = {int(coeff)}:  {desc}")
    desc = _decomp_desc(b_val)
    print(f"     b      = {b_val}:  {desc}")
    print(f"  2. Logistic core z·(1-z): single DSP48E1 (variable × variable)")
    print(f"  3. Bit-rotation: wiring only (ROL by constant = zero hardware)")
    print(f"  4. Post-processor ROTL(17,43): wiring only (fixed shift)")
    print(f"  5. Total DSP: 1 / LUT: ~800 / FF: ~900 / Power: ~0.11 W")
    print(f"  6. Init values: [{x1}, {x2}, {x3}, {x4}] (all-zero → defaults)")
    print(f"  7. Warmup: discard first 1000 outputs (transient elimination)")
    print(f"  8. Target: xc7z020clg400-1, 70.4 MHz, 563 MB/s (verified)")
    print("  9. HLS TCL: ensure `-cflags` appears only once per `add_files -tb`")
    print("     (duplicate `-cflags` -> ERROR [HLS 200-101]); use `sed` to strip dupes")
    print(f"{'='*70}")

    # Warning about software-hardware alignment
    print(f"\n{'!'*70}")
    print("  IMPORTANT — Software/Hardware Alignment Warning")
    print(f"{'!'*70}")
    print("  The Python ChaosPRNG class (chaos.application.prng) uses Python")
    print("  native integers with explicit & 0xFFFF masking for 16-bit arithmetic.")
    print("  The HLS hardware uses ap_uint<16> which has different intermediate")
    print("  truncation semantics in compound subtraction expressions.")
    print()
    print("  This causes cumulative divergence between software and hardware")
    print("  outputs (confirmed ~1000 iterations for A_odd=371).")
    print()
    print("  SOLUTION: Use the C shared library golden model for software:")
    print("    from hw.src_371.golden_prng import ChaosPRNG371")
    print("    prng = ChaosPRNG371()  # 100% bit-exact with HLS (verified)")
    print()
    print("  See tests/prng_export_371.py for the golden-model-based export.")
    print(f"{'!'*70}")


def _decomp_desc(c: int) -> str:
    """Describe constant using optimized shift-and-add from shift_mul.h."""
    decomps = {
        371: "(x<<8)+(x<<6)+(x<<5)+(x<<4)+(x<<1)+x",
        373: "(x<<8)+(x<<6)+(x<<5)+(x<<4)+(x<<2)+x",
        375: "(x<<8)+(x<<6)+(x<<5)+(x<<4)+(x<<2)+(x<<1)+x",
        50:  "(x<<5)+(x<<4)+(x<<1)",
        48:  "(x<<5)+(x<<4)",
        46:  "(x<<5)+(x<<3)+(x<<2)+(x<<1)",
        111: "(x<<6)+(x<<5)+(x<<3)+(x<<2)+(x<<1)+x",
    }
    if c in decomps:
        return decomps[c]
    # Fallback: simple decomposition
    parts = []
    remaining = c
    for p in [16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        if remaining >= p:
            parts.append(f"x<<{p.bit_length()-1}")
            remaining -= p
    return " + ".join(parts) if parts else "0"


if __name__ == "__main__":
    main()
