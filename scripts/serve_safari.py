#!/usr/bin/env python3
"""Serve a small local SAFARI web app for EDF staging."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import shutil
import tempfile
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from predict_safari_lgbm_v0 import predict_edf, write_outputs


STATE_LABELS = {"w": "Wake", "n": "NREM", "r": "REM"}


def safe_relpath(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def find_edfs(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".edf", ".bdf"}])


def slug_name(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return out.strip("._") or "record"


def render_page(title: str, body: str) -> bytes:
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17201b;
      --muted: #5d6a63;
      --line: #d9e0dc;
      --bg: #f7f8f5;
      --panel: #ffffff;
      --accent: #1f6f57;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 20px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 16px;
      margin-bottom: 22px;
    }}
    h1 {{ font-size: 26px; margin: 0; letter-spacing: 0; }}
    h2 {{ font-size: 18px; margin: 26px 0 10px; letter-spacing: 0; }}
    p {{ margin: 8px 0; color: var(--muted); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
    }}
    th {{ font-size: 12px; text-transform: uppercase; color: var(--muted); }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{
      background: #edf1ee;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin: 16px 0 20px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 12px;
    }}
    .metric strong {{ display: block; font-size: 20px; }}
    .actions {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    button, .button {{
      appearance: none;
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      padding: 7px 10px;
      border-radius: 5px;
      text-decoration: none;
      cursor: pointer;
      font: inherit;
    }}
    .button.secondary {{
      color: var(--accent);
      background: white;
    }}
    .warn {{
      border: 1px solid #d7b56d;
      background: #fff8e5;
      padding: 12px;
      margin: 14px 0;
    }}
    .error {{
      border: 1px solid #d7877e;
      background: #fff0ee;
      padding: 12px;
      margin: 14px 0;
    }}
    .small {{ font-size: 13px; color: var(--muted); }}
    @media (max-width: 720px) {{
      table, thead, tbody, tr, th, td {{ display: block; }}
      thead {{ display: none; }}
      tr {{ border-bottom: 1px solid var(--line); }}
      td {{ border-bottom: 0; padding: 8px 10px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>SAFARI</h1>
    <span class="small">Species-Agnostic Framework for Automated sleep scoRIng</span>
  </header>
  {body}
</main>
</body>
</html>"""
    return page.encode("utf-8")


class SafariServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], args: argparse.Namespace):
        super().__init__(server_address, handler_class)
        self.data_dir = args.data_dir.resolve()
        self.output_dir = args.output_dir.resolve()
        self.model = args.model.resolve()
        self.epoch_s = float(args.epoch_s)
        self.allow_uploads = bool(args.allow_uploads)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class SafariHandler(BaseHTTPRequestHandler):
    server: SafariServer

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_html(self, body: str, status: int = 200, title: str = "SAFARI") -> None:
        payload = render_page(title, body)
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def parse_form(self) -> dict[str, list[str]]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return urllib.parse.parse_qs(raw)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.render_index()
        elif parsed.path == "/result":
            self.render_result(urllib.parse.parse_qs(parsed.query).get("record", [""])[0])
        elif parsed.path.startswith("/download/"):
            self.send_download(parsed.path.removeprefix("/download/"))
        else:
            self.send_html("<div class='error'>Not found.</div>", status=404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/predict":
            form = self.parse_form()
            self.handle_predict(form.get("record", [""])[0])
        else:
            self.send_html("<div class='error'>Not found.</div>", status=404)

    def render_index(self) -> None:
        rows = []
        for edf in find_edfs(self.server.data_dir):
            rel = safe_relpath(edf, self.server.data_dir)
            size_mb = edf.stat().st_size / 1024 / 1024
            rel_q = html.escape(rel, quote=True)
            rows.append(
                "<tr>"
                f"<td><code>{html.escape(rel)}</code></td>"
                f"<td>{size_mb:.1f} MB</td>"
                "<td class='actions'>"
                "<form method='post' action='/predict'>"
                f"<input type='hidden' name='record' value='{rel_q}'>"
                "<button type='submit'>Stage</button>"
                "</form>"
                f"<a class='button secondary' href='/result?record={urllib.parse.quote(rel)}'>Result</a>"
                "</td>"
                "</tr>"
            )
        if rows:
            table = "<table><thead><tr><th>Recording</th><th>Size</th><th>Action</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        else:
            table = "<div class='warn'>No EDF/BDF files found in the mounted data directory.</div>"

        model_msg = ""
        if not self.server.model.exists():
            model_msg = f"<div class='error'>Model file not found: <code>{html.escape(str(self.server.model))}</code></div>"
        body = f"""
<h2>Local Staging</h2>
<p>Data directory: <code>{html.escape(str(self.server.data_dir))}</code></p>
<p>Output directory: <code>{html.escape(str(self.server.output_dir))}</code></p>
<p>Model: <code>{html.escape(str(self.server.model))}</code></p>
{model_msg}
{table}
<h2>Docker Mounts</h2>
<p>Mount EDF files at <code>/data</code>, the model at <code>/models/safari_lgbm_v0.txt</code>, and outputs at <code>/outputs</code>.</p>
"""
        self.send_html(body)

    def handle_predict(self, rel: str) -> None:
        try:
            edf = (self.server.data_dir / rel).resolve()
            edf.relative_to(self.server.data_dir)
            if not edf.exists() or not edf.is_file():
                raise FileNotFoundError(edf)
            if not self.server.model.exists():
                raise FileNotFoundError(self.server.model)
            stem = slug_name(Path(rel).with_suffix("").as_posix())
            csv_path = self.server.output_dir / "predictions" / f"{stem}.csv"
            npz_path = self.server.output_dir / "predictions" / f"{stem}.npz"
            result = predict_edf(edf, self.server.model, self.server.epoch_s)
            write_outputs(result, csv_path, npz_path)
            meta_path = self.server.output_dir / "predictions" / f"{stem}.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "record": rel,
                        "source_edf": str(edf),
                        "csv": str(csv_path),
                        "npz": str(npz_path),
                        "model": str(self.server.model),
                        "epoch_s": self.server.epoch_s,
                        "channel_names": result["channel_names"],
                        "channel_map": result["channel_map"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.send_redirect(f"/result?record={urllib.parse.quote(rel)}")
        except Exception as exc:
            self.send_html(f"<div class='error'><strong>Prediction failed.</strong><br>{html.escape(str(exc))}</div>", status=500)

    def result_paths(self, rel: str) -> tuple[Path, Path, Path]:
        stem = slug_name(Path(rel).with_suffix("").as_posix())
        base = self.server.output_dir / "predictions"
        return base / f"{stem}.csv", base / f"{stem}.npz", base / f"{stem}.json"

    def render_result(self, rel: str) -> None:
        if not rel:
            self.send_redirect("/")
            return
        csv_path, npz_path, meta_path = self.result_paths(rel)
        if not csv_path.exists():
            self.send_html(
                f"<div class='warn'>No prediction found for <code>{html.escape(rel)}</code>.</div><p><a class='button secondary' href='/'>Back</a></p>",
                status=404,
            )
            return
        df = pd.read_csv(csv_path)
        fractions = df["prediction"].value_counts(normalize=True).reindex(["w", "n", "r"], fill_value=0.0)
        minutes = df["prediction"].value_counts().reindex(["w", "n", "r"], fill_value=0) * self.server.epoch_s / 60.0
        mean_conf = float(df["confidence"].mean())
        metrics = []
        for key in ["w", "n", "r"]:
            metrics.append(
                f"<div class='metric'><span>{STATE_LABELS[key]}</span><strong>{fractions[key] * 100:.1f}%</strong>"
                f"<span class='small'>{minutes[key]:.1f} min</span></div>"
            )
        rows = []
        for row in df.head(20).itertuples(index=False):
            rows.append(
                "<tr>"
                f"<td>{int(row.epoch)}</td><td>{float(row.time_s):.1f}</td><td>{html.escape(str(row.prediction))}</td>"
                f"<td>{float(row.confidence):.3f}</td><td>{float(row.p_wake):.3f}</td>"
                f"<td>{float(row.p_nrem):.3f}</td><td>{float(row.p_rem):.3f}</td>"
                "</tr>"
            )
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        body = f"""
<h2>Result</h2>
<p>Recording: <code>{html.escape(rel)}</code></p>
<div class="meta">
  {''.join(metrics)}
  <div class='metric'><span>Mean confidence</span><strong>{mean_conf:.3f}</strong><span class='small'>{len(df)} epochs</span></div>
</div>
<div class="actions">
  <a class="button" href="/download/{urllib.parse.quote(safe_relpath(csv_path, self.server.output_dir))}">Download CSV</a>
  <a class="button secondary" href="/download/{urllib.parse.quote(safe_relpath(npz_path, self.server.output_dir))}">Download NPZ</a>
  <a class="button secondary" href="/">Back</a>
</div>
<h2>First 20 Epochs</h2>
<table><thead><tr><th>Epoch</th><th>Time s</th><th>Stage</th><th>Conf</th><th>P Wake</th><th>P NREM</th><th>P REM</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Channels</h2>
<pre>{html.escape(json.dumps(meta.get("channel_map", {}), indent=2))}</pre>
"""
        self.send_html(body)

    def send_download(self, rel: str) -> None:
        try:
            rel = urllib.parse.unquote(rel)
            path = (self.server.output_dir / rel).resolve()
            path.relative_to(self.server.output_dir)
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(path)
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile)
        except Exception as exc:
            self.send_html(f"<div class='error'>Download failed: {html.escape(str(exc))}</div>", status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/outputs"))
    parser.add_argument("--model", type=Path, default=Path("/models/safari_lgbm_v0.txt"))
    parser.add_argument("--epoch-s", type=float, default=4.0)
    parser.add_argument("--allow-uploads", action="store_true", help="Reserved for a future upload UI")
    args = parser.parse_args()

    if args.host == "127.0.0.1" and os.environ.get("SAFARI_DOCKER"):
        args.host = "0.0.0.0"
    httpd = SafariServer((args.host, args.port), SafariHandler, args)
    print(f"SAFARI listening on http://{args.host}:{args.port}")
    print(f"Data: {httpd.data_dir}")
    print(f"Model: {httpd.model}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
