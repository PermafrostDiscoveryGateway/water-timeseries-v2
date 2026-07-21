"""Tests for ``repartition_parquet`` — correctness and row-group pruning.

The dashboard's per-lake reads rely on the NRT vector parquet being sorted by
``id_geohash`` with small row groups so a ``id_geohash == <id>`` filter can
skip all but the one small row group containing that lake. These tests verify
the rewrite preserves data (including GeoParquet metadata) and actually makes
pruning possible.
"""

import random

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import Polygon

from water_timeseries.scripts.repartition_parquet import (
    _widen_binary_columns,
    repartition_parquet,
)

SEED = 7
N_ROWS = 6000


def _rand_geohash(rng: random.Random) -> str:
    alphabet = "0123456789bcdefghjkmnpqrstuvwxyz"
    return "".join(rng.choice(alphabet) for _ in range(12))


def _build_gdf(n: int = N_ROWS) -> gpd.GeoDataFrame:
    """Synthetic GeoParquet shaped like the NRT lake file, shuffled (unsorted)."""
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    ids = list({_rand_geohash(rng) for _ in range(int(n * 1.1))})[:n]
    n = len(ids)
    lons = np_rng.uniform(-165, -140, n)
    lats = np_rng.uniform(60, 70, n)
    geoms = [
        Polygon([(x, y), (x + 0.001, y), (x + 0.001, y + 0.001), (x, y + 0.001)])
        for x, y in zip(lons, lats)
    ]
    gdf = gpd.GeoDataFrame(
        {
            "id_geohash": ids,
            "water_residual": np_rng.normal(0, 1, n),
            "date_break_year": np_rng.integers(2017, 2026, n),
            "geometry": geoms,
        },
        crs="EPSG:4326",
    )
    return gdf.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def _row_groups_matching(path, lake_id: str) -> int:
    """How many row groups a ``id_geohash == lake_id`` filter must scan."""
    md = pq.read_metadata(path)
    col_idx = md.schema.names.index("id_geohash")
    matched = 0
    for rg in range(md.num_row_groups):
        stats = md.row_group(rg).column(col_idx).statistics
        if stats is None or stats.min is None:
            matched += 1  # no stats -> unprunable
        elif stats.min <= lake_id <= stats.max:
            matched += 1
    return matched


@pytest.fixture
def original_parquet(tmp_path):
    """Unsorted parquet written as one large row group (like the real file)."""
    gdf = _build_gdf()
    path = tmp_path / "original.parquet"
    gdf.to_parquet(path, row_group_size=N_ROWS)  # single big row group
    return path, gdf


def test_output_is_sorted(original_parquet, tmp_path):
    src, _ = original_parquet
    out = repartition_parquet(src, tmp_path / "repart.parquet", row_group_size=500)
    ids = pq.read_table(out, columns=["id_geohash"]).column("id_geohash").to_pylist()
    assert ids == sorted(ids)


def test_small_row_groups_created(original_parquet, tmp_path):
    src, _ = original_parquet
    out = repartition_parquet(src, tmp_path / "repart.parquet", row_group_size=500)
    md = pq.read_metadata(out)
    assert md.num_row_groups == pytest.approx(N_ROWS / 500, abs=1)
    assert md.num_row_groups > pq.read_metadata(src).num_row_groups


def test_no_data_loss(original_parquet, tmp_path):
    src, gdf = original_parquet
    out = repartition_parquet(src, tmp_path / "repart.parquet", row_group_size=500)
    result = gpd.read_parquet(out)
    assert len(result) == len(gdf)
    assert set(result["id_geohash"]) == set(gdf["id_geohash"])
    # values preserved (compare after aligning on id_geohash)
    a = gdf.set_index("id_geohash").sort_index()
    b = result.set_index("id_geohash").sort_index()
    np.testing.assert_allclose(a["water_residual"], b["water_residual"])


def test_geoparquet_metadata_preserved(original_parquet, tmp_path):
    """Output must stay readable by geopandas with intact geometry + CRS."""
    src, gdf = original_parquet
    out = repartition_parquet(src, tmp_path / "repart.parquet", row_group_size=500)
    result = gpd.read_parquet(out)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == gdf.crs
    assert result.geometry.notna().all()
    assert (~result.geometry.is_empty).all()


def test_pruning_enabled_after_repartition(original_parquet, tmp_path):
    """The core perf property: a per-lake read scans ~1 row group, not all."""
    src, gdf = original_parquet
    out = repartition_parquet(src, tmp_path / "repart.parquet", row_group_size=500)

    rng = random.Random(SEED)
    sample = [rng.choice(gdf["id_geohash"].tolist()) for _ in range(50)]

    src_md = pq.read_metadata(src)
    orig_scan = np.mean([_row_groups_matching(src, i) for i in sample])
    repart_scan = np.mean([_row_groups_matching(out, i) for i in sample])

    # Original: single unsorted group -> every read scans the whole file.
    assert orig_scan == src_md.num_row_groups
    # Repartitioned: unique ids sorted -> each read scans essentially one group.
    assert repart_scan <= 1.5


def test_filtered_read_returns_correct_lake(original_parquet, tmp_path):
    src, gdf = original_parquet
    out = repartition_parquet(src, tmp_path / "repart.parquet", row_group_size=500)
    target = gdf["id_geohash"].iloc[len(gdf) // 2]
    result = gpd.read_parquet(out, filters=[("id_geohash", "==", target)])
    assert len(result) == 1
    assert result.iloc[0]["id_geohash"] == target


def test_widen_binary_columns_casts_to_large_variants():
    """binary/string columns become large_binary/large_string; metadata kept.

    Guards the fix for the real-file failure: ``sort_by`` concatenates the WKB
    geometry column, and a 32-bit ``binary`` column over 2 GB overflows with
    ``ArrowInvalid: offset overflow``. Widening to 64-bit offsets avoids it.
    """
    table = pa.table(
        {"id_geohash": ["a", "b"], "wkb": [b"\x00\x01", b"\x02"]},
        metadata={b"geo": b"{}"},
    )
    assert pa.types.is_string(table.schema.field("id_geohash").type)
    assert pa.types.is_binary(table.schema.field("wkb").type)

    widened = _widen_binary_columns(table)

    assert pa.types.is_large_string(widened.schema.field("id_geohash").type)
    assert pa.types.is_large_binary(widened.schema.field("wkb").type)
    assert widened.schema.metadata == {b"geo": b"{}"}  # GeoParquet key preserved
    assert widened.column("wkb").to_pylist() == [b"\x00\x01", b"\x02"]  # data intact


def test_output_geometry_is_large_binary(original_parquet, tmp_path):
    """Repartitioned geometry uses large_binary and stays geopandas-readable."""
    src, _ = original_parquet
    out = repartition_parquet(src, tmp_path / "repart.parquet", row_group_size=500)
    schema = pq.read_schema(out)
    geom_field = schema.field("geometry")
    assert pa.types.is_large_binary(geom_field.type)
    # still decodes as real geometry
    assert gpd.read_parquet(out).geometry.notna().all()


def test_rejects_in_place_rewrite(original_parquet):
    src, _ = original_parquet
    with pytest.raises(ValueError, match="must differ"):
        repartition_parquet(src, src)


def test_rejects_missing_sort_column(original_parquet, tmp_path):
    src, _ = original_parquet
    with pytest.raises(ValueError, match="not found"):
        repartition_parquet(src, tmp_path / "repart.parquet", sort_column="nope")
