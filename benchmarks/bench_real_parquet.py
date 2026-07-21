"""Benchmark per-lake reads on a REAL parquet file: original vs repartitioned.

Unlike ``bench_repartition.py`` (synthetic, cache-resident), this runs on an
actual file so the numbers reflect real geometry sizes and row-group layout.

Usage:
    python benchmarks/bench_real_parquet.py <input.parquet> [--out <repart.parquet>]
                                            [--row-group-size 2000] [--n 30]

If --out does not exist it is generated once via repartition_parquet.
"""

import argparse
import random
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq

from water_timeseries.scripts.repartition_parquet import repartition_parquet


def bytes_per_read(path: Path, lake_id: str) -> tuple[int, int]:
    """(row groups, compressed bytes) a `id_geohash == lake_id` filter must fetch."""
    md = pq.read_metadata(path)
    ci = md.schema.names.index("id_geohash")
    groups = nbytes = 0
    for rg in range(md.num_row_groups):
        m = md.row_group(rg)
        s = m.column(ci).statistics
        if s is None or s.min is None or (s.min <= lake_id <= s.max):
            groups += 1
            nbytes += m.total_byte_size
    return groups, nbytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--row-group-size", type=int, default=2000)
    ap.add_argument("--n", type=int, default=30, help="random lakes to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    orig = args.input
    repart = args.out or Path(tempfile.gettempdir()) / (orig.stem + "_repartitioned.parquet")

    if not repart.exists():
        print(f"Repartitioning -> {repart} (one-time)...")
        t0 = time.perf_counter()
        repartition_parquet(orig, repart, row_group_size=args.row_group_size)
        print(f"  took {time.perf_counter() - t0:.1f}s")

    om, rm = pq.read_metadata(orig), pq.read_metadata(repart)
    print("\n--- File layout ---")
    print(f"ORIGINAL      : {om.num_rows:,} rows, {om.num_row_groups} row groups, {orig.stat().st_size/1e9:.2f} GB")
    print(f"REPARTITIONED : {rm.num_rows:,} rows, {rm.num_row_groups} row groups, {repart.stat().st_size/1e9:.2f} GB")

    ids = pq.read_table(repart, columns=["id_geohash"]).column("id_geohash").to_pylist()
    rng = random.Random(args.seed)
    sample = [rng.choice(ids) for _ in range(args.n)]

    o = [bytes_per_read(orig, i) for i in sample]
    r = [bytes_per_read(repart, i) for i in sample]
    o_b, r_b = np.mean([x[1] for x in o]), np.mean([x[1] for x in r])
    print("\n--- Row groups scanned / read ---")
    print(f"ORIGINAL      : {np.mean([x[0] for x in o]):.1f} / {om.num_row_groups}")
    print(f"REPARTITIONED : {np.mean([x[0] for x in r]):.2f} / {rm.num_row_groups}")
    print("\n--- Bytes fetched / read (drives cold-read latency) ---")
    print(f"ORIGINAL      : {o_b/1e6:,.1f} MB")
    print(f"REPARTITIONED : {r_b/1e6:,.3f} MB")
    print(f"REDUCTION     : {o_b/r_b:,.0f}x fewer bytes per lake read")

    print("\n--- Wall-clock (warm) ---")
    n_o = min(5, args.n)
    t0 = time.perf_counter()
    for i in sample[:n_o]:
        gpd.read_parquet(orig, filters=[("id_geohash", "==", i)])
    o_t = (time.perf_counter() - t0) / n_o
    t0 = time.perf_counter()
    for i in sample:
        gpd.read_parquet(repart, filters=[("id_geohash", "==", i)])
    r_t = (time.perf_counter() - t0) / len(sample)
    print(f"ORIGINAL      : {o_t*1000:,.0f} ms/read (avg of {n_o})")
    print(f"REPARTITIONED : {r_t*1000:,.1f} ms/read (avg of {len(sample)})")
    print(f"SPEEDUP       : {o_t/r_t:,.0f}x")


if __name__ == "__main__":
    main()
