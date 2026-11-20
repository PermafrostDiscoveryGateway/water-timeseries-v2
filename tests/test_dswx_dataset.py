"""Tests for the OPERA DSWx-S1 dataset handler (DSWxDataset) and its registration."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from water_timeseries.dataset import DSWxDataset, create_water_dataset
from water_timeseries.utils.data import (
    DSWX_WATER_TOTAL,
    dswx_bandnames,
    dswx_plot_variables,
    get_water_dataset_type,
)

LAKE = "b7g678chrqb5"
DATES = ("2025-06-01", "2025-07-01", "2025-08-01")
DEFAULTS = {
    "area_open_water": [[1.0, 2.0, 3.0]],
    "area_inundated_vegetation": [[0.5, 0.5, 0.5]],
    "area_not_water": [[18.0, 17.0, 16.0]],
}


def _ds(values=None, dates=DATES, id_name="id_geohash", lake=LAKE) -> xr.Dataset:
    values = values or DEFAULTS
    return xr.Dataset(
        {v: ((id_name, "date"), np.array(vals, dtype=float)) for v, vals in values.items()},
        coords={id_name: [lake], "date": pd.to_datetime(dates)},
    )


class TestDSWxRegistration:
    """The third source must be registered in the single dispatch point."""

    def test_sniffer_identifies_dswx(self):
        assert get_water_dataset_type(_ds()) == "dswx"

    def test_sniffer_does_not_confuse_the_other_sources(self):
        dw = xr.Dataset({"water": (("id_geohash", "date"), np.zeros((1, 2)))})
        jrc = xr.Dataset({"area_water_permanent": (("id_geohash", "date"), np.zeros((1, 2)))})
        assert get_water_dataset_type(dw) == "dynamic_world"
        assert get_water_dataset_type(jrc) == "jrc"
        assert get_water_dataset_type(_ds()) != "dynamic_world"

    def test_unknown_variables_raise_a_helpful_error(self):
        empty = xr.Dataset({"something_else": (("id_geohash", "date"), np.zeros((1, 2)))})
        with pytest.raises(ValueError, match="area_open_water"):
            get_water_dataset_type(empty)

    def test_factory_returns_dswx_dataset(self):
        assert isinstance(create_water_dataset(_ds()), DSWxDataset)

    def test_factory_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown water dataset type"):
            create_water_dataset(_ds(), "not_a_source")


class TestDSWxDataset:
    """Schema, derived variables, masking and identifiers."""

    def test_water_column_and_data_columns(self):
        ds = DSWxDataset(_ds())
        assert ds.water_column == "area_open_water"
        assert ds.data_columns == dswx_bandnames

    def test_total_water_is_the_union_of_both_water_classes(self):
        ds = DSWxDataset(_ds())
        assert ds.ds[DSWX_WATER_TOTAL].values[0].tolist() == [1.5, 2.5, 3.5]

    def test_area_data_and_nodata_are_derived(self):
        ds = DSWxDataset(_ds())
        assert ds.ds["area_data"].values[0].tolist() == [19.5, 19.5, 19.5]
        assert ds.ds["area_nodata"].values[0].tolist() == [0.0, 0.0, 0.0]

    def test_plot_columns_include_total_and_components_only(self):
        ds = DSWxDataset(_ds())
        assert ds.plot_columns == dswx_plot_variables
        assert DSWX_WATER_TOTAL in ds.plot_columns
        assert "area_not_water" not in ds.plot_columns
        for column in ds.plot_columns:
            assert column in ds.ds.data_vars

    def test_nodata_tolerance_decides_masking(self):
        # third date keeps 94.6 % of the best observation -> 5.4 % no-data
        lossy = {
            "area_open_water": [[1.0, 2.0, 2.7]],
            "area_inundated_vegetation": [[0.5, 0.5, 0.45]],
            "area_not_water": [[18.0, 17.0, 15.3]],
        }
        strict = DSWxDataset(_ds(lossy))
        assert np.isnan(strict.ds["area_open_water"].values[0][2])

        tolerant = DSWxDataset(_ds(lossy), max_nodata_fraction=0.10)
        assert not np.isnan(tolerant.ds["area_open_water"].values[0][2])

    def test_invalid_nodata_tolerance_raises(self):
        with pytest.raises(ValueError, match="max_nodata_fraction"):
            DSWxDataset(_ds(), max_nodata_fraction=1.5)

    def test_mask_data_false_keeps_everything(self):
        lossy = {
            "area_open_water": [[1.0, 2.0, 2.7]],
            "area_inundated_vegetation": [[0.5, 0.5, 0.45]],
            "area_not_water": [[18.0, 17.0, 15.3]],
        }
        ds = DSWxDataset(_ds(lossy), mask_data=False)
        assert not np.isnan(ds.ds["area_open_water"].values[0][2])

    def test_custom_id_field(self):
        ds = DSWxDataset(_ds(id_name="lake_id"), id_field="lake_id")
        assert ds.object_ids_ == [LAKE]
        assert ds.water_column == "area_open_water"

    def test_unknown_id_field_raises(self):
        # In practice validation happens during construction: _normalize_ds already
        # needs the identifier dimension, so the ValueError surfaces in __init__.
        with pytest.raises(ValueError, match="does not have"):
            DSWxDataset(_ds(), id_field="not_a_field")

    def test_plots_render_and_accept_the_object_id_kwarg(self):
        ds = DSWxDataset(_ds(), mask_data=False)
        assert ds.plot_timeseries(LAKE) is not None
        assert ds.plot_timeseries_interactive(object_id=LAKE) is not None
        with pytest.warns(FutureWarning, match="object_id"):
            ds.plot_timeseries(id_geohash=LAKE)
