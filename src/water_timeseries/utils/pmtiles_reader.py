"""Read metadata from PMTiles v3 archives without loading tile data."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any


def read_pmtiles_header(path: Path | str) -> dict[str, Any]:
    """Parse the fixed 127-byte PMTiles v3 header (bounds, zoom, center).

    See https://github.com/protomaps/PMTiles/blob/main/spec/v3/spec.md
    """
    path = Path(path)
    with path.open("rb") as fh:
        header = fh.read(127)
    if len(header) != 127 or header[:7] != b"PMTiles":
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
