"""Read metadata from PMTiles v3 archives without loading tile data."""

from __future__ import annotations

import json
import struct
import urllib.request
from pathlib import Path
from typing import Any


def read_pmtiles_header(path: Path | str) -> dict[str, Any]:
    """Parse the fixed 127-byte PMTiles v3 header (bounds, zoom, center).

    See https://github.com/protomaps/PMTiles/blob/main/spec/v3/spec.md
    """
    path = Path(path)
    with path.open("rb") as fh:
        header = fh.read(127)
    if len(header) < 127 or header[:7] != b"PMTiles":
        raise ValueError(f"Not a PMTiles v3 file: {path}")
    if header[7] != 0x3:
        raise ValueError(f"Unsupported PMTiles version byte: {header[7]}")

    min_zoom = header[100]
    max_zoom = header[101]
    min_lon_e7, min_lat_e7, max_lon_e7, max_lat_e7 = struct.unpack_from("<iiii", header, 102)
    center_zoom = header[118]
    center_lon_e7, center_lat_e7 = struct.unpack_from("<ii", header, 119)

    e7 = 1e7
    min_lon, min_lat = min_lon_e7 / e7, min_lat_e7 / e7
    max_lon, max_lat = max_lon_e7 / e7, max_lat_e7 / e7
    center_lon, center_lat = center_lon_e7 / e7, center_lat_e7 / e7

    return {
        "min_zoom": min_zoom,
        "max_zoom": max_zoom,
        "center_zoom": center_zoom,
        "bounds": [[min_lon, min_lat], [max_lon, max_lat]],
        "center": [center_lon, center_lat],
        "zoom": center_zoom if center_zoom else max(2, min_zoom),
    }


def read_pmtiles_header_remote(url: str) -> dict[str, Any]:
    """Fetch PMTiles header from a remote URL using a range request."""
    # Request bytes 0 through 126 (which is exactly 127 bytes)
    req = urllib.request.Request(url, headers={"Range": "bytes=0-126"})
    with urllib.request.urlopen(req) as resp:
        header = resp.read()

    if len(header) < 127 or header[:7] != b"PMTiles":
        raise ValueError(f"Not a valid PMTiles v3 remote URL: {url}")

    # Ensure we only process the first 127 bytes in case a server sends more
    header = header[:127]

    # PMTiles v3 header layout uses Int32 scaled by 1e7 starting at byte 102
    min_zoom = header[100]
    max_zoom = header[101]
    min_lon_e7, min_lat_e7, max_lon_e7, max_lat_e7 = struct.unpack_from("<iiii", header, 102)
    center_zoom = header[118]
    center_lon_e7, center_lat_e7 = struct.unpack_from("<ii", header, 119)

    e7 = 1e7
    min_lon, min_lat = min_lon_e7 / e7, min_lat_e7 / e7
    max_lon, max_lat = max_lon_e7 / e7, max_lat_e7 / e7
    center_lon, center_lat = center_lon_e7 / e7, center_lat_e7 / e7

    return {
        "min_zoom": min_zoom,
        "max_zoom": max_zoom,
        "center_zoom": center_zoom,
        "bounds": [[min_lon, min_lat], [max_lon, max_lat]],
        "center": [center_lon, center_lat],
        "zoom": center_zoom if center_zoom else max(2, min_zoom),
    }


def _metadata_extent(header: bytes) -> tuple[int, int, int]:
    """Return ``(offset, length, internal_compression)`` of the JSON metadata block.

    The PMTiles v3 header stores the metadata offset/length at bytes 24-39 and
    the compression applied to it (and to the directories) at byte 97: 1 is
    "none", 2 is gzip. See the spec linked in ``read_pmtiles_header``.
    """
    offset, length = struct.unpack_from("<QQ", header, 24)
    return offset, length, header[97]


def _decode_metadata(raw: bytes, internal_compression: int) -> dict[str, Any]:
    if internal_compression == 2:
        import gzip

        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def read_pmtiles_metadata(path: Path | str) -> dict[str, Any]:
    """Parse the JSON metadata block of a PMTiles v3 archive.

    This is what tippecanoe writes about the build itself -- ``vector_layers``,
    ``tilestats``, the per-zoom ``strategies`` it had to fall back on, and the
    command line it ran. ``archive_bakes_low_zoom_centroids`` reads it to tell
    whether an archive's centroid layer survived the build.
    """
    path = Path(path)
    with path.open("rb") as fh:
        header = fh.read(127)
        if len(header) < 127 or header[:7] != b"PMTiles":
            raise ValueError(f"Not a PMTiles v3 file: {path}")
        offset, length, compression = _metadata_extent(header)
        fh.seek(offset)
        raw = fh.read(length)
    return _decode_metadata(raw, compression)


def read_pmtiles_metadata_remote(url: str) -> dict[str, Any]:
    """Fetch a remote archive's JSON metadata with two range requests."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-126"})
    with urllib.request.urlopen(req) as resp:
        header = resp.read()[:127]
    if len(header) < 127 or header[:7] != b"PMTiles":
        raise ValueError(f"Not a valid PMTiles v3 remote URL: {url}")

    offset, length, compression = _metadata_extent(header)
    if length == 0:
        return {}
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()[:length]
    return _decode_metadata(raw, compression)
