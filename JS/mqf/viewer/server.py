"""A small HTTP server on 127.0.0.1: the page, the arrays it draws, and the videos (with Range
requests, so the browser can seek inside a two-hour file). Standard library only.

Nothing is reachable from another machine: the socket is bound to the loopback address.
"""
from __future__ import annotations

import json
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

STATIC = Path(__file__).parent / "static"
STATIC_FILES = {"/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/app.css": ("app.css", "text/css; charset=utf-8"),
                "/favicon.svg": ("favicon.svg", "image/svg+xml")}
CHUNK = 1 << 20
# The browser drops connections all the time (every seek cancels the video request it had open);
# on Windows that surfaces as WinError 10053/10054. Expected, so never printed.
CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, port: int, meta: dict, blobs: dict[str, bytes], videos: dict[str, Path]):
        self.meta_json = json.dumps(meta).encode()
        self.blobs = blobs
        self.videos = {k: Path(v) for k, v in videos.items()}
        super().__init__(("127.0.0.1", port), Handler)

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], CLIENT_GONE):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: ViewerServer

    def log_message(self, *args):          # quiet: the terminal shows the viewer's own log
        pass

    def do_HEAD(self):
        self._route(head=True)

    def do_GET(self):
        self._route(head=False)

    # ------------------------------------------------------------------------------------------
    def _route(self, head: bool):
        path = unquote(urlparse(self.path).path)
        try:
            if path in STATIC_FILES:
                name, ctype = STATIC_FILES[path]
                return self._bytes((STATIC / name).read_bytes(), ctype, head)
            if path == "/api/meta":
                return self._bytes(self.server.meta_json, "application/json", head)
            m = re.fullmatch(r"/api/bin/([A-Za-z0-9_]+)", path)
            if m and m.group(1) in self.server.blobs:
                return self._bytes(self.server.blobs[m.group(1)], "application/octet-stream", head)
            m = re.fullmatch(r"/video/([a-z0-9_]+)", path)
            if m and m.group(1) in self.server.videos:
                return self._file(self.server.videos[m.group(1)], "video/mp4", head)
            self._status(HTTPStatus.NOT_FOUND)
        except CLIENT_GONE:
            pass                            # the browser cancelled (seeking does this all the time)

    def _status(self, code: HTTPStatus, extra: dict | None = None):
        self.send_response(code)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _bytes(self, data: bytes, ctype: str, head: bool):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def _file(self, path: Path, ctype: str, head: bool):
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range", "")
        partial = rng.startswith("bytes=")
        if partial:
            first = rng[6:].split(",")[0].strip()
            a, _, b = first.partition("-")
            try:
                if a == "":
                    start, end = max(0, size - int(b)), size - 1
                else:
                    start, end = int(a), (int(b) if b else size - 1)
            except ValueError:
                return self._status(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, {"Content-Range": f"bytes */{size}"})
            if start >= size or start > end:
                return self._status(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, {"Content-Range": f"bytes */{size}"})
            end = min(end, size - 1)
        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head:
            return
        with open(path, "rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                buf = fh.read(min(CHUNK, left))
                if not buf:
                    break
                self.wfile.write(buf)
                left -= len(buf)


def start(meta: dict, blobs: dict[str, bytes], videos: dict[str, Path], port: int = 8765,
          tries: int = 20) -> ViewerServer:
    """Bind to the first free port from `port` on."""
    last = None
    for p in range(port, port + tries):
        try:
            return ViewerServer(p, meta, blobs, videos)
        except OSError as e:
            last = e
    raise SystemExit(f"no free port in {port}..{port + tries - 1}: {last}")
