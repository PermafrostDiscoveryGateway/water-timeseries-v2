"""Tests for bbox subsetting in ``breakpoint-analysis-nrt``.

Covers ``resolve_lake_subset`` in ``water_timeseries.scripts.cli``:
- the guard that any ``--bbox-*`` requires ``--vector-file``;
- centroid-box filtering via the shared ``filter_gdf_by_bbox`` helper
  (same semantics as the historical command);
- partial boxes, no-bbox behaviour, and the missing-column error.
"""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from water_timeseries.scripts.cli import resolve_lake_subset


@pytest.fixture
def lakes_vector(tmp_path: Path) -> Path:
    """Five point lakes on a regular grid, written to a GeoParquet file."""
    xs = [100.0, 105.0, 110.0, 115.0, 120.0]  # lon
    ys = [20.0, 25.0, 30.0, 35.0, 40.0]  # lat
    g = gpd.GeoDataFrame(
        {
            "id_geohash": [f"lk{i}" for i in range(5)],
            "geometry": [Point(x, y) for x, y in zip(xs, ys)],
        },
        crs="EPSG:4326",
    )
    vec = tmp_path / "lakes.parquet"
    g.to_parquet(vec, index=False)
    return vec


class TestBboxRequiresVector:
    def test_bbox_without_vector_file_exits(self):
        with pytest.raises(SystemExit) as exc:
            resolve_lake_subset(None, bbox_west=100)
        assert exc.value.code == 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bbox_south": 20},
            {"bbox_east": 110, "bbox_north": 30},
        ],
    )
    def test_any_single_bbox_boundary_triggers_guard(self, kwargs):
        with pytest.raises(SystemExit) as exc:
            resolve_lake_subset(None, **kwargs)
        assert exc.value.code == 1


class TestVectorWithoutBbox:
    def test_returns_all_ids(self, lakes_vector):
        assert sorted(resolve_lake_subset(lakes_vector)) == ["lk0", "lk1", "lk2", "lk3", "lk4"]

    def test_no_vector_no_bbox_returns_none(self):
        assert resolve_lake_subset(None) is None

    def test_missing_id_geohash_column_exits(self, tmp_path):
        g = gpd.GeoDataFrame({"label_id": [1], "geometry": [Point(1, 1)]}, crs="EPSG:4326")
        vec = tmp_path / "no_geohash.parquet"
        g.to_parquet(vec, index=False)
        with pytest.raises(SystemExit) as exc:
            resolve_lake_subset(vec)
        assert exc.value.code == 1


class TestBboxFiltering:
    def test_full_box_keeps_all(self, lakes_vector):
        ids = resolve_lake_subset(lakes_vector, bbox_west=100, bbox_south=20, bbox_east=120, bbox_north=40)
        assert sorted(ids) == ["lk0", "lk1", "lk2", "lk3", "lk4"]

    def test_west_and_east_partial_box(self, lakes_vector):
        ids = resolve_lake_subset(lakes_vector, bbox_west=105, bbox_east=115)
        assert sorted(ids) == ["lk1", "lk2", "lk3"]

    def test_single_south_boundary(self, lakes_vector):
        assert sorted(resolve_lake_subset(lakes_vector, bbox_south=30)) == ["lk2", "lk3", "lk4"]

    def test_single_north_boundary(self, lakes_vector):
        assert sorted(resolve_lake_subset(lakes_vector, bbox_north=25)) == ["lk0", "lk1"]

    def test_empty_box_returns_no_ids(self, lakes_vector):
        assert resolve_lake_subset(lakes_vector, bbox_west=200, bbox_east=300) == []
