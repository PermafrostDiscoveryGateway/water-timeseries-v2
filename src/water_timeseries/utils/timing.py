"""Lightweight timing helper for profiling dashboard hot paths.

Every block times itself and logs the elapsed time via loguru. Callers that
want to surface the numbers in the Streamlit UI (rather than just the log
file/console) can pass a ``sink`` dict, which is where ``map_viewer.py``
accumulates timings for the sidebar debug expander.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger


@contextmanager
def timed(label: str, sink: dict[str, float] | None = None) -> Iterator[None]:
    """Time the wrapped block, logging ``label: N.N ms`` and optionally recording it.

    Args:
        label: Human-readable name for the timed block, e.g. ``"build_pmtiles_map"``.
        sink: If given, ``sink[label]`` is set to the elapsed time in milliseconds.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"[timing] {label}: {elapsed_ms:.1f} ms")
        if sink is not None:
            sink[label] = elapsed_ms
