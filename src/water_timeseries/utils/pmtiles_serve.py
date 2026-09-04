"""Serve PMTiles and the MapLibre map page over HTTP (Range requests required for PMTiles)."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self
from urllib.parse import parse_qs, unquote, urlparse

from loguru import logger

from water_timeseries.utils.pmtiles_build import POINT_POLY_SWITCH_ZOOM

_MAP_HTML = Path(__file__).parent.parent / "dashboard" / "static" / "lake_map.html"


def build_map_url(base_url: str, config: dict) -> str:
    """Return a /map URL with the full config encoded as a base64 query parameter."""
    import base64
    import json

    b64 = base64.urlsafe_b64encode(json.dumps(config).encode()).decode().rstrip("=")
    return f"{base_url.rstrip('/')}/map?config={b64}"


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

        prefix = getattr(self.server, "path_prefix", "")
        if prefix and route.startswith(prefix):
            route = route[len(prefix) :]
        if not route.startswith("/"):
            route = "/" + route

        if route in ("/map", "/map.html"):
            self._serve_map_page(parse_qs(parsed.query), head_only=head_only)
            return

        self._serve_file(route.lstrip("/"), head_only=head_only)

    def _serve_map_page(self, query: dict[str, list[str]], head_only: bool = False) -> None:
        config_id = (query.get("config_id") or [None])[0]
        config_b64 = (query.get("config") or [None])[0]

        if config_id and hasattr(self.server, "config_cache") and config_id in self.server.config_cache:
            config = self.server.config_cache[config_id]
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

        # Only fill in the pmtiles_url when the config didn't bring its own
        # (mounted archives resolve theirs up front, per session, in
        # PmtilesServer.pmtiles_url_for).
        pmtiles_name = getattr(self.server, "pmtiles_filename", None)
        if pmtiles_name and not config.get("pmtiles_url"):
            config["pmtiles_url"] = f"{self.server.base_url}/{pmtiles_name}"  # type: ignore[attr-defined]

        # The page gates its circle/polygon layers on this rather than carrying
        # its own copy of the number the tiles were baked with -- but only when
        # the config says the archive bakes centroids below it. Callers that
        # know the archive set that (see pmtiles_viewer._build_map_config); the
        # default here is the reading that keeps lakes on screen for every
        # archive built before those centroids existed.
        config.setdefault("point_poly_switch_zoom", POINT_POLY_SWITCH_ZOOM)
        config.setdefault("base_has_centroids", False)

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

        # Archives mounted via register_pmtiles() live under /<token>/<name>.
        directory = None
        mounts = getattr(self.server, "mounts", None) or {}
        token, _, remainder = rel_path.partition("/")
        if remainder and token in mounts:
            directory = mounts[token]
            rel_path = remainder
        # In HTML-only remote URL mode, the server won't have a static file directory mapped.
        elif getattr(self.server, "directory", None):
            directory = self.server.directory  # type: ignore[attr-defined]

        if directory is None:
            self.send_error(404, "Not found")
            return

        path = (directory / rel_path).resolve()
        directory = directory.resolve()
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
        pmtiles_file: Path | str | None = None,
        host: str = "0.0.0.0",
        port: int = 0,
        public_host: str | None = None,
    ):
        if pmtiles_file is not None:
            path = Path(pmtiles_file).resolve()
            if path.is_dir():
                self.directory = path
                self.pmtiles_path = None
                self.pmtiles_filename = None
            else:
                self.pmtiles_path = path
                self.directory = path.parent
                self.pmtiles_filename = path.name
        else:
            self.pmtiles_path = None
            self.directory = _MAP_HTML.parent  # Fallback map location
            self.pmtiles_filename = None

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
        env_base_url = os.environ.get("PMTILES_BASE_URL")
        self.path_prefix = urlparse(env_base_url).path.rstrip("/") if env_base_url else ""
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.config_cache: dict[str, Any] = {}
        # Extra directories mounted under /<token>/ by register_pmtiles().
        self.mounts: dict[str, Path] = {}

    def register_pmtiles(self, pmtiles_file: Path | str) -> str:
        """Mount an extra .pmtiles archive and return its URL path.

        One process serves every browser session, and switching dashboard modes
        swaps the tileset -- so archives are *added* rather than swapped in
        place: a session still viewing the other mode keeps working. Each
        parent directory gets a short mount token, which also keeps two
        same-named archives in different directories apart.
        """
        path = Path(pmtiles_file).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PMTiles file not found: {path}")

        parent = path.parent
        token = next((tok for tok, mounted in self.mounts.items() if mounted == parent), None)
        if token is None:
            token = f"t{len(self.mounts)}"
            self.mounts[token] = parent
            logger.info(f"PMTiles server mounted {parent} at /{token}")
        return f"{token}/{path.name}"

    def pmtiles_url_for(self, pmtiles_file: Path | str) -> str:
        """Absolute URL the browser should fetch this archive from."""
        return f"{self.base_url}/{self.register_pmtiles(pmtiles_file)}"

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("Server is not running")
        # PMTILES_BASE_URL overrides the full base URL — use when the server is
        # behind a reverse proxy (e.g. nginx ingress) that rewrites the path.
        # Example: https://example.com/pmtiles
        override = os.environ.get("PMTILES_BASE_URL")
        if override:
            return override.rstrip("/")
        return f"http://{self.public_host}:{self._httpd.server_port}"

    def start(self) -> PmtilesServer:
        self._httpd = ThreadingHTTPServer((self.host, self.port), _PmtilesHTTPRequestHandler)
        self._httpd.directory = self.directory
        self._httpd.pmtiles_filename = self.pmtiles_filename
        self._httpd.config_cache = self.config_cache
        self._httpd.mounts = self.mounts
        self._httpd.path_prefix = self.path_prefix
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

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()
