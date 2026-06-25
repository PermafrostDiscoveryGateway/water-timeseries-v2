"""Serve PMTiles and the MapLibre map page over HTTP (Range requests required for PMTiles)."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from loguru import logger

_MAP_HTML = Path(__file__).parent.parent / "dashboard" / "static" / "lake_map.html"


class _PmtilesHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler: map page + static files with byte-range support for .pmtiles."""

    def log_message(self, format: str, *args) -> None:
        logger.info(f"PMTiles Server: {format % args}")

    def do_OPTIONS(self) -> None:
        logger.debug(f"CORS preflight request: {self.path}")
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET(head_only=True)

    def do_GET(self, head_only: bool = False) -> None:
        parsed = urlparse(self.path)
        route = unquote(parsed.path)

        if route in ("/map", "/map.html"):
            self._serve_map_page(parse_qs(parsed.query), head_only=head_only)
            return

        self._serve_file(route.lstrip("/"), head_only=head_only)

    def _serve_map_page(self, query: dict[str, list[str]], head_only: bool = False) -> None:
        config_id = (query.get("config_id") or [None])[0]
        config_b64 = (query.get("config") or [None])[0]

        if config_id and hasattr(self.server, "config_cache") and config_id in getattr(self.server, "config_cache"):
            config = getattr(self.server, "config_cache")[config_id]
        elif config_b64:
            try:
                padding = "=" * (-len(config_b64) % 4)
                config = json.loads(base64.urlsafe_b64decode(config_b64 + padding).decode("utf-8"))
            except (json.JSONDecodeError, ValueError) as exc:
                self.send_error(400, f"Invalid config: {exc}")
                return
        else:
            self.send_error(400, "Missing config or config_id query parameter")
            return

        pmtiles_name = self.server.pmtiles_filename  # type: ignore[attr-defined]
        config["pmtiles_url"] = f"{self.server.base_url}/{pmtiles_name}"  # type: ignore[attr-defined]

        template = _MAP_HTML.read_text(encoding="utf-8")
        html = template.replace("__CONFIG_JSON__", json.dumps(config))

        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _stream_file(self, fh: Path, start: int, length: int, head_only: bool) -> None:
        if head_only or length <= 0:
            return
        with open(fh, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def _serve_file(self, rel_path: str, head_only: bool = False) -> None:
        if not rel_path:
            self.send_error(404, "Not found")
            return

        logger.info(f"Serving file: {rel_path}")
        path = (self.server.directory / rel_path).resolve()  # type: ignore[attr-defined]
        directory = self.server.directory.resolve()  # type: ignore[attr-defined]
        if not str(path).startswith(str(directory)) or not path.is_file():
            logger.warning(f"File not found: {rel_path}")
            self.send_error(404, "Not found")
            return

        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")

        if range_header and range_header.startswith("bytes="):
            logger.debug(f"Range request for {rel_path}: {range_header}")
            range_spec = range_header.removeprefix("bytes=").split(",", 1)[0]
            start_s, _, end_s = range_spec.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                self.send_error(416, "Range not satisfiable")
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self._send_cors()
            self.end_headers()
            self._stream_file(path, start, length, head_only)
            return

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self._send_cors()
        self.end_headers()
        self._stream_file(path, 0, size, head_only)

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")


class PmtilesServer:
    """Background HTTP server for a .pmtiles file and the MapLibre map page (same origin)."""

    def __init__(
        self,
        pmtiles_file: Path | str,
        host: str = "0.0.0.0",
        port: int = 0,
        public_host: Optional[str] = None,
    ):
        self.pmtiles_path = Path(pmtiles_file).resolve()
        self.directory = self.pmtiles_path.parent
        self.pmtiles_filename = self.pmtiles_path.name
        self.host = host
        # Allow fixed port via env var (required for Docker port publishing).
        # Default 0 = OS picks a random free port (works for local uv runs).
        env_port = os.environ.get("PMTILES_PORT")
        self.port = int(env_port) if env_port else port
        # public_host is the hostname the *browser* should use to reach this
        # server.  When running in Docker the container binds to 0.0.0.0 but
        # the browser must use "localhost" (or the host IP) via the published
        # port.  Override with the PMTILES_HOST env var or pass explicitly.
        self.public_host: str = public_host or os.environ.get("PMTILES_HOST", "localhost")
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.config_cache: dict[str, Any] = {}

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("Server is not running")
        # Use public_host so the URL is reachable from the browser, even when
        # the server socket is bound to 0.0.0.0 inside a Docker container.
        return f"http://{self.public_host}:{self._httpd.server_port}"

    def start(self) -> "PmtilesServer":
        self._httpd = ThreadingHTTPServer((self.host, self.port), _PmtilesHTTPRequestHandler)
        self._httpd.directory = self.directory
        self._httpd.pmtiles_filename = self.pmtiles_filename
        self._httpd.config_cache = self.config_cache
        self._httpd.base_url = ""  # set after bind
        self.port = self._httpd.server_port
        self._httpd.base_url = self.base_url
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"PMTiles server started at {self.base_url}")
        logger.info(f"Serving files from: {self.directory}")
        logger.info(f"PMTiles file: {self.pmtiles_filename}")
        return self

    def map_iframe_url(self, config: dict[str, Any]) -> str:
        """URL for ``st.iframe`` — map HTML and PMTiles share this origin."""
        import uuid

        config_id = uuid.uuid4().hex
        self.config_cache[config_id] = config

        # Clean up old configs to prevent memory leak (keep last 5)
        if len(self.config_cache) > 5:
            keys_to_remove = list(self.config_cache.keys())[:-5]
            for k in keys_to_remove:
                self.config_cache.pop(k, None)

        return f"{self.base_url}/map?config_id={config_id}"

    def stop(self) -> None:
        logger.info("Stopping PMTiles server")
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def url_for(self, filename: str) -> str:
        return f"{self.base_url}/{Path(filename).name}"

    def __enter__(self) -> "PmtilesServer":
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()
