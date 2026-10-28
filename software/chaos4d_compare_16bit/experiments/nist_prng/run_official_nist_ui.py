# Maintainer: zhaowenbo
"""Simple local browser UI for the bundled official NIST SP 800-22 suite."""

from __future__ import annotations

import argparse
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import warnings
from urllib.parse import parse_qs
import sys

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="cgi")
    import cgi

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from metrics.nist_randomness import DEFAULT_NIST_SEQUENCE_BIT_LENGTH, DEFAULT_NIST_SEQUENCE_COUNT
from utils.official_nist_runner import (
    list_official_nist_tests,
    run_official_nist_assess,
)


DEFAULT_NIST_ROOT = Path(__file__).resolve().parents[2] / "nist_sp_800_22"


def _html_page(body: str) -> bytes:
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Official NIST SP 800-22 UI</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,sans-serif;margin:2rem;line-height:1.45;}"
        ".layout{display:grid;grid-template-columns:minmax(360px,1.1fr) minmax(340px,0.9fr);gap:1.25rem;align-items:start;}"
        ".panel{border:1px solid #ddd;border-radius:10px;padding:1rem;background:#fff;}"
        ".result-panel{min-height:260px;}"
        "form{padding:0;max-width:none;border:none;border-radius:0;}"
        "label{display:block;font-weight:600;margin-top:0.9rem;}"
        "input,select,textarea{width:100%;padding:0.55rem;margin-top:0.25rem;box-sizing:border-box;}"
        "button{margin-top:1rem;padding:0.7rem 1rem;}"
        "pre{background:#f7f7f7;padding:1rem;overflow:auto;border-radius:8px;white-space:pre-wrap;word-break:break-word;}"
        "h1,h2,h3{margin-top:0;}"
        ".note{color:#555;margin:0.3rem 0 1rem;}"
        ".ok{color:#0a6b2d;font-weight:600;}"
        ".warn{color:#8a5a00;font-weight:600;}"
        ".status{margin-bottom:1rem;}"
        ".inline-row{display:flex;gap:0.75rem;align-items:end;}"
        ".inline-row > *{flex:1;}"
        ".inline-row button{flex:0 0 auto;white-space:nowrap;}"
        "@media (max-width: 820px){.layout{grid-template-columns:1fr;}}"
        "</style></head><body>"
        f"{body}</body></html>"
    ).encode("utf-8")


def _test_options_html(selected_tests: list[int] | None = None) -> str:
    selected = set(selected_tests or [test_id for test_id, _ in list_official_nist_tests()])
    rows: list[str] = []
    for test_id, test_name in list_official_nist_tests():
        checked = " checked" if test_id in selected else ""
        rows.append(
            "<label style='font-weight:400;display:flex;gap:0.5rem;align-items:center;'>"
            f"<input type='checkbox' name='selected_tests' value='{test_id}'{checked} style='width:auto;margin:0;' />"
            f"<span>[{test_id:02d}] {escape(test_name)}</span>"
            "</label>"
        )
    return "\n".join(rows)


def _default_result_panel() -> str:
    return (
        "<h2>Command Output</h2>"
        "<p class='note'>Run the official suite and the execution summary will appear here. "
        "The panel shows the launched command, automatic parameters, output paths, and the tail of the command-line log.</p>"
    )


def _waiting_result_panel() -> str:
    return (
        "<h2>Command Output</h2>"
        "<p class='warn'>Official NIST SP 800-22 is running. Please wait patiently.</p>"
        "<p class='note'>The official suite may take some time to finish, especially for large inputs and multiple test categories. "
        "Please do not refresh or close the page while the current request is being processed.</p>"
    )


def _choose_output_directory(initial_dir: str) -> str:
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(initialdir=initial_dir or ".", title="Select output folder")
    finally:
        root.destroy()
    return str(selected).strip()


def _render_form(
    message: str = "",
    result_html: str = "",
    *,
    source_mode: str = "upload_file",
    custom_input_file: str = "",
    input_mode: str = "ascii",
    output_root: str = "results/official_nist_ui",
    selected_tests: list[int] | None = None,
) -> bytes:
    body = f"""
    <h1>Official NIST SP 800-22 UI</h1>
    <p class="note">Manual items only: choose the data source, input type, test categories, and output location. The UI keeps the remaining settings automatic: bundled <code>nist_sp_800_22/assess</code>, sequence length <code>{DEFAULT_NIST_SEQUENCE_BIT_LENGTH}</code>, and sequence count <code>{DEFAULT_NIST_SEQUENCE_COUNT}</code>.</p>
    <div class="layout">
      <div class="panel">
        {'<p class="ok status">' + escape(message) + '</p>' if message else ''}
        <form method="post" action="/run" enctype="multipart/form-data" id="nist-form">
          <label>Data source</label>
          <select name="source_mode">
            <option value="upload_file"{' selected' if source_mode == 'upload_file' else ''}>Choose a file from the operating system file picker</option>
            <option value="existing_file"{' selected' if source_mode == 'existing_file' else ''}>Use an existing file path directly</option>
          </select>

          <label>Choose local file</label>
          <input type="file" name="uploaded_file" />
          <div class="note">Used when the source mode is <code>upload_file</code>. The selected file is saved under <code>nist_sp_800_22/ui_inputs/</code>.</div>

          <label>Existing input file</label>
          <input name="custom_input_file" value="{escape(custom_input_file)}" placeholder="nist_sp_800_22/ui_inputs/my_bits.txt" />
          <div class="note">Used when the source mode is <code>existing_file</code>. Prefer a no-space path under <code>nist_sp_800_22/</code> because the official parser is space-sensitive.</div>

          <label>Data type</label>
          <select name="input_mode">
            <option value="ascii"{' selected' if input_mode == 'ascii' else ''}>ASCII 0/1 stream</option>
            <option value="binary"{' selected' if input_mode == 'binary' else ''}>Binary bytes</option>
          </select>

          <label>Test categories</label>
          <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0.3rem 1rem;padding:0.8rem;border:1px solid #ddd;border-radius:8px;">
            {_test_options_html(selected_tests)}
          </div>

          <label>Result output location</label>
          <div class="inline-row">
            <input name="output_root" value="{escape(output_root)}" readonly />
            <button type="submit" name="action" value="choose_output_dir">Choose Folder</button>
          </div>
          <div class="note">Choose an operating-system folder. A new timestamped run directory will be created under this location automatically.</div>

          <button type="submit" name="action" value="run" id="run-button">Run Official Assess</button>
        </form>
      </div>
      <div class="panel result-panel" id="result-panel">
        {result_html or _default_result_panel()}
      </div>
    </div>
    <script>
      (function() {{
        const form = document.getElementById('nist-form');
        const resultPanel = document.getElementById('result-panel');
        if (!form || !resultPanel) return;
        form.addEventListener('submit', function(event) {{
          const submitter = event.submitter;
          if (!submitter || submitter.value !== 'run') return;
          resultPanel.innerHTML = `{_waiting_result_panel()}`;
          submitter.disabled = true;
          submitter.textContent = 'Running...';
        }});
      }})();
    </script>
    """
    return _html_page(body)


def _read_log_tail(log_path: Path, max_lines: int = 80) -> str:
    if not log_path.exists():
        return "Log file not found yet."
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(["..."] + lines[-max_lines:])


class OfficialNISTUIHandler(BaseHTTPRequestHandler):
    @staticmethod
    def _coerce_form_value(item: object) -> object:
        if isinstance(item, list):
            coerced: list[object] = []
            for value in item:
                if getattr(value, "filename", None):
                    coerced.append(value)
                else:
                    coerced.append(value.value)
            return coerced
        if getattr(item, "filename", None):
            return item
        return item.value

    @staticmethod
    def _as_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]

    def _parse_form(self) -> dict[str, object]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )
            data: dict[str, object] = {}
            for key in form.keys():
                data[key] = self._coerce_form_value(form[key])
            return data

        content_length = int(self.headers.get("Content-Length", "0"))
        parsed = parse_qs(self.rfile.read(content_length).decode("utf-8"))
        data: dict[str, object] = {}
        for key, values in parsed.items():
            data[key] = values if len(values) > 1 else values[0]
        return data

    def _send_html_response(self, body: bytes, status: HTTPStatus = HTTPStatus.OK) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, socket.error):
            return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if self.path not in {"/", "/run"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_html_response(_render_form())

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        form_data = self._parse_form()
        source_mode = str(form_data.get("source_mode", "upload_file")).strip() or "upload_file"
        custom_input_file = str(form_data.get("custom_input_file", "")).strip()
        input_mode = str(form_data.get("input_mode", "ascii")).strip().lower() or "ascii"
        output_root = Path(str(form_data.get("output_root", "results/official_nist_ui")).strip() or "results/official_nist_ui")
        selected_tests = [int(value) for value in self._as_list(form_data.get("selected_tests"))]
        action = str(form_data.get("action", "run"))

        try:
            if action == "choose_output_dir":
                selected_output_dir = _choose_output_directory(str(output_root))
                if selected_output_dir:
                    output_root = Path(selected_output_dir)
                    message = f"Output folder selected: {output_root}"
                else:
                    message = "Output folder selection cancelled."
                self._send_html_response(
                    _render_form(
                        message=message,
                        source_mode=source_mode,
                        custom_input_file=custom_input_file,
                        input_mode=input_mode,
                        output_root=str(output_root),
                        selected_tests=selected_tests or None,
                    )
                )
                return

            if not selected_tests:
                raise ValueError("Please select at least one NIST test category.")

            nist_root = DEFAULT_NIST_ROOT
            sequence_bit_length = DEFAULT_NIST_SEQUENCE_BIT_LENGTH
            sequence_count = DEFAULT_NIST_SEQUENCE_COUNT
            local_input_dir = nist_root / "ui_inputs"
            local_input_dir.mkdir(parents=True, exist_ok=True)
            input_path: Path
            if source_mode == "upload_file":
                uploaded_item = form_data.get("uploaded_file")
                if uploaded_item is None or not getattr(uploaded_item, "file", None):
                    raise ValueError("upload_file mode requires choosing a local file.")
                uploaded_name = Path(str(getattr(uploaded_item, "filename", "uploaded_bits.txt"))).name or "uploaded_bits.txt"
                file_bytes = uploaded_item.file.read()
                input_path = local_input_dir / uploaded_name
                input_path.write_bytes(file_bytes)
                source_note = f"Uploaded file saved to {input_path}"
            elif source_mode == "existing_file":
                if not custom_input_file:
                    raise ValueError("existing_file mode requires an input file path.")
                input_path = Path(custom_input_file)
                source_note = f"Existing input file selected directly: {input_path}"
            else:
                raise ValueError(f"unsupported source_mode: {source_mode}")

            result_parts = [
                "<h2>Command Output</h2>",
                "<h3>Prepared Input</h3>",
                "<pre>"
                + escape(
                    "\n".join(
                        [
                            source_note,
                            f"Input mode: {input_mode}",
                            f"Selected tests: {', '.join(str(test_id) for test_id in selected_tests)}",
                            f"Sequence bit length: {sequence_bit_length}",
                            f"Sequence count: {sequence_count}",
                            f"Output root: {output_root}",
                        ]
                    )
                )
                + "</pre>",
            ]
            if action == "run":
                label = f"{input_path.stem}_{input_mode}"
                assess_command = f"{nist_root / 'assess'} {sequence_bit_length}"
                artifacts = run_official_nist_assess(
                    nist_root=nist_root,
                    input_file=input_path,
                    sequence_bit_length=sequence_bit_length,
                    sequence_count=sequence_count,
                    label=label,
                    output_root=output_root,
                    input_mode=input_mode,
                    selected_tests=selected_tests,
                )
                stdout_log_path = Path(artifacts.stdout_log)
                result_parts.extend(
                    [
                        "<h3>Execution Summary</h3>",
                        "<pre>"
                        + escape(
                            "\n".join(
                                [
                                    f"Command: {assess_command}",
                                    f"Working directory: {nist_root}",
                                    f"Run dir: {artifacts.run_dir}",
                                    f"Original input file: {artifacts.input_file}",
                                    f"Staged input file: {artifacts.staged_input_file}",
                                    f"Runtime NIST root: {artifacts.runtime_nist_root}",
                                    f"Input mode: {input_mode}",
                                    f"Selected tests: {', '.join(str(test_id) for test_id in selected_tests)}",
                                    f"Final report: {artifacts.final_report}",
                                    f"Runtime AlgorithmTesting dir: {artifacts.algorithm_testing_dir}",
                                    f"Summary JSON: {artifacts.summary_json}",
                                    f"STDOUT log: {artifacts.stdout_log}",
                                ]
                            )
                        )
                        + "</pre>",
                        "<h3>Command-Line Log Tail</h3>",
                        "<pre>" + escape(_read_log_tail(stdout_log_path)) + "</pre>",
                    ]
                )

            self._send_html_response(
                _render_form(
                    message="Request completed.",
                    result_html="\n".join(result_parts),
                    source_mode=source_mode,
                    custom_input_file=custom_input_file,
                    input_mode=input_mode,
                    output_root=str(output_root),
                    selected_tests=selected_tests,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._send_html_response(
                _render_form(
                    message=f"Error: {exc}",
                    source_mode=source_mode,
                    custom_input_file=custom_input_file,
                    input_mode=input_mode,
                    output_root=str(output_root),
                    selected_tests=selected_tests or None,
                )
            )

    def log_error(self, format: str, *args: object) -> None:  # noqa: A003
        if args and isinstance(args[0], BrokenPipeError):
            return
        super().log_error(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a simple local UI for the bundled official NIST SP 800-22 suite.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--list-datasets", action="store_true")
    args = parser.parse_args()

    if args.list_datasets:
        for test_id, test_name in list_official_nist_tests():
            print(f"{test_id:02d}: {test_name}")
        return

    server = ThreadingHTTPServer((args.host, args.port), OfficialNISTUIHandler)
    print(f"Official NIST UI listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
