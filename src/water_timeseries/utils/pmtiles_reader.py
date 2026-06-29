"""Read metadata from PMTiles v3 archives without loading tile data."""

from __future__ import annotations

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
