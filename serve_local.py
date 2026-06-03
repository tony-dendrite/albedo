#!/usr/bin/env python3
"""Combined local dev server for the albedo dashboard + demo data.

Serves on http://localhost:3000

Routing:
  /html/*         → website/html/
  /css/*          → website/css/
  /js/*           → website/js/
  /dashboard.json → demo_data_hippus/dashboard.json
  /evals/*        → demo_data_hippus/evals/
  /               → redirect to /html/index.html
"""
import http.server
import os
import urllib.parse
from pathlib import Path

PORT = 3000
ROOT = Path(__file__).parent

ROUTES = {
    "/html":       ROOT / "website" / "html",
    "/css":        ROOT / "website" / "css",
    "/js":         ROOT / "website" / "js",
    "/evals":      ROOT / "demo_data_hippus" / "evals",
    "/dashboard.json": ROOT / "demo_data_hippus" / "dashboard.json",
}

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jsonl":"application/x-ndjson; charset=utf-8",
    ".gz":   "application/gzip",
    ".svg":  "image/svg+xml",
    ".ico":  "image/x-icon",
    ".png":  "image/png",
    ".woff2":"font/woff2",
}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/" or path == "":
            self._redirect("/html/index.html")
            return

        # Exact file match first (dashboard.json)
        if path in ROUTES and ROUTES[path].is_file():
            self._send_file(ROUTES[path])
            return

        # Prefix match for directories
        for prefix, base in ROUTES.items():
            if not base.is_dir():
                continue
            if path.startswith(prefix + "/") or path == prefix:
                rel = path[len(prefix):].lstrip("/")
                target = base / rel if rel else base
                if target.is_file():
                    self._send_file(target)
                    return
                if target.is_dir():
                    idx = target / "index.html"
                    if idx.exists():
                        self._send_file(idx)
                        return
                break

        self._send_404()

    def _redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _send_file(self, path: Path):
        try:
            data = path.read_bytes()
        except OSError:
            self._send_404()
            return
        ext  = path.suffix.lower()
        mime = MIME.get(ext, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_404(self):
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"not found")

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")


if __name__ == "__main__":
    print(f"Dashboard: http://localhost:{PORT}/html/index.html")
    print(f"Detail:    http://localhost:{PORT}/html/detail.html")
    print(f"Data:      http://localhost:{PORT}/dashboard.json")
    with http.server.HTTPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
