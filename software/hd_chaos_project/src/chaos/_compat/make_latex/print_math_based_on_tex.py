#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Backward-compatible wrapper.
The canonical LaTeX compilation utilities remain here (unchanged) for IPython/Spyder workflows.
"""

import sys
import os, re, shutil, subprocess, tempfile
from pathlib import Path
from typing import Optional, Tuple
from IPython.display import display, IFrame, HTML


MINIMAL_TEMPLATE = r"""\documentclass[preview,border=2pt]{standalone}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{bm}
\usepackage{graphicx}
\begin{document}
%s
\end{document}
"""


def show_pdf(path, width=900, height=700, try_image=True, open_external_if_needed=True):
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    try:
        from IPython.display import display, IFrame, HTML
        url = p.as_uri()
        try:
            display(IFrame(src=url, width=width, height=height))
            return p
        except Exception:
            html = f'<embed src="{url}" type="application/pdf" width="{width}" height="{height}">'
            display(HTML(html))
            return p
    except Exception:
        pass
    if try_image:
        try:
            from pdf2image import convert_from_path
            imgs = convert_from_path(str(p), dpi=200, first_page=1, last_page=1)
            if imgs:
                try:
                    from IPython.display import display
                    display(imgs[0])
                    return p
                except Exception:
                    pass
        except Exception:
            pass
    if open_external_if_needed:
        try:
            if sys.platform.startswith("darwin"):
                subprocess.run(["open", str(p)], check=False)
            elif os.name == "nt":
                os.startfile(str(p))
            else:
                subprocess.run(["xdg-open", str(p)], check=False)
        except Exception:
            print(f"[WARN] Cannot open viewer: {p}")
    print(f"PDF: {p}")
    return p


def _looks_like_full_doc(tex: str) -> bool:
    return r"\documentclass" in tex or r"\begin{document}" in tex


def compile_tex(tex_source: str,
                out_dir: Optional[str] = None,
                engine_priority=("latexmk", "pdflatex", "xelatex", "lualatex"),
                out_fmt: str = "pdf",
                jobname: str = "output") -> Tuple[Path, str]:
    assert out_fmt in ("pdf", "png"), "out_fmt must be 'pdf' or 'png'"
    workdir = Path(tempfile.mkdtemp(prefix="texbuild_"))
    tex_path = workdir / f"{jobname}.tex"
    is_full = _looks_like_full_doc(tex_source)
    tex_content = tex_source if is_full else (MINIMAL_TEMPLATE % tex_source)
    tex_path.write_text(tex_content, encoding="utf-8")

    def which(cmd): return shutil.which(cmd) is not None
    chosen = None
    for eng in engine_priority:
        if which(eng):
            chosen = eng
            break
    if chosen is None:
        raise RuntimeError("No LaTeX compiler found.")

    log = []
    try:
        if chosen == "latexmk":
            cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
        else:
            cmd = [chosen, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
        p = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.append(f"$ {' '.join(cmd)}\n{p.stdout}")
        if p.returncode != 0:
            raise RuntimeError("LaTeX compilation failed.")

        pdf_path = workdir / f"{jobname}.pdf"
        if not pdf_path.exists():
            candidates = list(workdir.glob("*.pdf"))
            if not candidates:
                raise RuntimeError("No PDF found.")
            pdf_path = candidates[0]

        out_path = pdf_path
        if out_fmt == "png":
            if not which("convert"):
                raise RuntimeError("ImageMagick `convert` required for PNG.")
            png_path = workdir / f"{jobname}.png"
            cmd2 = ["convert", "-density", "300", str(pdf_path) + "[0]", "-quality", "95", str(png_path)]
            p2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log.append(f"$ {' '.join(cmd2)}\n{p2.stdout}")
            if p2.returncode != 0 or not png_path.exists():
                raise RuntimeError("PDF->PNG conversion failed.")
            out_path = png_path

        if out_dir is None:
            final = Path.cwd() / out_path.name
        else:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            final = Path(out_dir) / out_path.name
        shutil.copy2(out_path, final)
        return final, "\n".join(log)
    finally:
        try:
            build_log = workdir / f"{jobname}.log"
            if build_log.exists():
                log_text = build_log.read_text(errors="ignore")
                log.append("\n===== .log =====\n" + log_text)
        except Exception:
            pass
        shutil.rmtree(workdir, ignore_errors=True)
