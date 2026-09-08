"""Tests for plotting functionality."""

import warnings

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest
import xarray as xr

from water_timeseries.breakpoint import SimpleBreakpoint
from water_timeseries.dataset import DWDataset, JRCDataset


def _renamed(dataset: xr.Dataset, new_id: str) -> xr.Dataset:
    """Return a copy of the dataset whose id dimension is renamed to `new_id`."""
    return dataset.rename({"id_geohash": new_id})


class TestDWDatasetPlotting:
    """Test DWDataset plotting functionality."""

    def test_dw_plot_timeseries_creates_figure(self, dw_test_dataset):
        """Test that plot_timeseries creates a matplotlib figure."""
        ds = DWDataset(dw_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        fig = ds.plot_timeseries(geohash)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dw_plot_has_water_line(self, dw_test_dataset):
        """Test that plot contains water data."""
        ds = DWDataset(dw_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        fig = ds.plot_timeseries(geohash)

        # Check that plot was created with data
        assert len(fig.axes) > 0
        ax = fig.axes[0]
        assert len(ax.lines) > 0 or len(ax.collections) > 0
        plt.close(fig)

    def test_dw_plot_with_different_geohashes(self, dw_test_dataset):
        """Test plotting with different geohash values."""
        ds = DWDataset(dw_test_dataset)
        geohashes = ds.ds.coords["id_geohash"].values

        for geohash in geohashes:
            fig = ds.plot_timeseries(geohash)
            assert fig is not None
            plt.close(fig)

    def test_dw_plot_timeseries_bp_date(self, dw_test_dataset):
        """Test that plot_timeseries creates a matplotlib figure."""
        ds = DWDataset(dw_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        fig = ds.plot_timeseries(geohash, breakpoints="2018-06-01")
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dw_plot_timeseries_bp_datelist(self, dw_test_dataset):
        """Test that plot_timeseries creates a matplotlib figure."""
        ds = DWDataset(dw_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        fig = ds.plot_timeseries(geohash, breakpoints=["2018-06-01", "2020-09-01"])
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dw_plot_timeseries_with_breakpoint_method(self, dw_test_dataset):
        """Test that plot_timeseries works with a BreakpointMethod."""
        ds = DWDataset(dw_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        bp = SimpleBreakpoint()
        fig = ds.plot_timeseries(geohash, breakpoints=bp)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dw_plot_timeseries_interactive_bp_date(self, dw_test_dataset):
        """Test that plot_timeseries_interactive creates a Plotly figure."""
        ds = DWDataset(dw_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        fig = ds.plot_timeseries_interactive(geohash, breakpoints="2018-06-01")
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_dw_plot_timeseries_interactive_datelist(self, dw_test_dataset):
        """Test that plot_timeseries_interactive creates a Plotly figure."""
        ds = DWDataset(dw_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        fig = ds.plot_timeseries_interactive(geohash, breakpoints=["2018-06-01", "2020-09-01"])
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_dw_plot_timeseries_interactive_with_breakpoint_method(self, dw_test_dataset):
        """Test that plot_timeseries_interactive works with a BreakpointMethod."""
        ds = DWDataset(dw_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        bp = SimpleBreakpoint()
        fig = ds.plot_timeseries_interactive(geohash, breakpoints=bp)
        assert fig is not None
        assert isinstance(fig, go.Figure)


class TestJRCDatasetPlotting:
    """Test JRCDataset plotting functionality."""

    def test_jrc_plot_timeseries_creates_figure(self, jrc_test_dataset):
        """Test that plot_timeseries creates a matplotlib figure."""
        ds = JRCDataset(jrc_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        fig = ds.plot_timeseries(geohash)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_jrc_plot_has_water_line(self, jrc_test_dataset):
        """Test that plot contains water data."""
        ds = JRCDataset(jrc_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        fig = ds.plot_timeseries(geohash)

        # Check that plot was created with data
        assert len(fig.axes) > 0
        ax = fig.axes[0]
        assert len(ax.lines) > 0 or len(ax.collections) > 0
        plt.close(fig)

    def test_jrc_plot_multiple_time_series(self, jrc_test_dataset):
        """Test that JRC plot shows multiple water types."""
        ds = JRCDataset(jrc_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        fig = ds.plot_timeseries(geohash)

        # Multiple lines/series should be plotted
        ax = fig.axes[0]
        # Should have lines for permanent, seasonal, and land
        assert len(ax.lines) >= 1 or len(ax.collections) >= 1
        plt.close(fig)

    def test_jrc_plot_timeseries_with_breakpoint_method(self, jrc_test_dataset):
        """Test that plot_timeseries works with a BreakpointMethod."""
        ds = JRCDataset(jrc_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        bp = SimpleBreakpoint()
        fig = ds.plot_timeseries(geohash, breakpoints=bp)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_jrc_plot_timeseries_interactive_bp_date(self, jrc_test_dataset):
        """Test that plot_timeseries_interactive creates a Plotly figure."""
        ds = JRCDataset(jrc_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        fig = ds.plot_timeseries_interactive(geohash, breakpoints="2018-06-01")
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_jrc_plot_timeseries_interactive_datelist(self, jrc_test_dataset):
        """Test that plot_timeseries_interactive creates a Plotly figure."""
        ds = JRCDataset(jrc_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        fig = ds.plot_timeseries_interactive(geohash, breakpoints=["2018-06-01", "2020-09-01"])
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_jrc_plot_timeseries_interactive_with_breakpoint_method(self, jrc_test_dataset):
        """Test that plot_timeseries_interactive works with a BreakpointMethod."""
        ds = JRCDataset(jrc_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        bp = SimpleBreakpoint()
        fig = ds.plot_timeseries_interactive(geohash, breakpoints=bp)
        assert fig is not None
        assert isinstance(fig, go.Figure)


class TestCustomFieldPlotting:
    """Ensure plotting works when the id column is named other than id_geohash (e.g. lake_id).

    The public entry points (``plot_timeseries`` / ``plot_timeseries_interactive``) take an
    opaque object identifier, so a dataset whose id coordinate is called ``lake_id`` must
    behave identically to one whose id coordinate is called ``id_geohash``.
    """

    def test_dw_plot_timeseries_lake_id(self, dw_test_dataset):
        """Static plot works for a dataset whose id column is ``lake_id``."""
        renamed = _renamed(dw_test_dataset, "lake_id")
        ds = DWDataset(renamed, id_field="lake_id")
        lake_id = ds.ds.coords["lake_id"].values[0]

        fig = ds.plot_timeseries(lake_id)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dw_plot_timeseries_interactive_lake_id(self, dw_test_dataset):
        """Interactive plot works for a dataset whose id column is ``lake_id``."""
        renamed = _renamed(dw_test_dataset, "lake_id")
        ds = DWDataset(renamed, id_field="lake_id")
        lake_id = ds.ds.coords["lake_id"].values[0]

        fig = ds.plot_timeseries_interactive(lake_id)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_jrc_plot_timeseries_lake_id(self, jrc_test_dataset):
        """Static plot works for a JRC dataset whose id column is ``lake_id``."""
        renamed = _renamed(jrc_test_dataset, "lake_id")
        ds = JRCDataset(renamed, id_field="lake_id")
        lake_id = ds.ds.coords["lake_id"].values[0]

        fig = ds.plot_timeseries(lake_id)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_jrc_plot_timeseries_interactive_lake_id(self, jrc_test_dataset):
        """Interactive plot works for a JRC dataset whose id column is ``lake_id``."""
        renamed = _renamed(jrc_test_dataset, "lake_id")
        ds = JRCDataset(renamed, id_field="lake_id")
        lake_id = ds.ds.coords["lake_id"].values[0]

        fig = ds.plot_timeseries_interactive(lake_id)
        assert fig is not None
        assert isinstance(fig, go.Figure)


class TestIdGeohashBackwardCompatibility:
    """Ensure the legacy ``id_geohash=`` keyword still works and emits a DeprecationWarning.

    All four plotting entry points (DW + JRC, static + interactive) must accept
    ``id_geohash=`` exactly as they did before the refactor, while also supporting
    the new ``object_id=`` keyword.
    """

    @pytest.mark.parametrize("method", ["plot_timeseries", "plot_timeseries_interactive"])
    def test_dw_id_geohash_kwarg_emits_deprecation_warning(self, dw_test_dataset, method):
        """DW static/interactive: legacy ``id_geohash=`` works but is deprecated."""
        ds = DWDataset(dw_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        with pytest.warns(DeprecationWarning, match="object_id"):
            fig = getattr(ds, method)(id_geohash=geohash)
        assert fig is not None
        if method == "plot_timeseries":
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        else:
            assert isinstance(fig, go.Figure)

    @pytest.mark.parametrize("method", ["plot_timeseries", "plot_timeseries_interactive"])
    def test_dw_object_id_kwarg_no_warning(self, dw_test_dataset, method):
        """DW static/interactive: new ``object_id=`` works without warning."""
        ds = DWDataset(dw_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)  # promote to error
            fig = getattr(ds, method)(object_id=geohash)
        assert fig is not None
        if method == "plot_timeseries":
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        else:
            assert isinstance(fig, go.Figure)

    def test_dw_both_kwargs_raises_type_error(self, dw_test_dataset):
        """Providing both ``object_id=`` and ``id_geohash=`` is an error."""
        ds = DWDataset(dw_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        with pytest.raises(TypeError, match="exactly one"):
            ds.plot_timeseries(object_id=geohash, id_geohash=geohash)

    def test_dw_neither_kwarg_raises_value_error(self, dw_test_dataset):
        """Providing neither ``object_id=`` nor ``id_geohash=`` is an error."""
        ds = DWDataset(dw_test_dataset)

        with pytest.raises(ValueError, match="At least one"):
            ds.plot_timeseries()

    @pytest.mark.parametrize("method", ["plot_timeseries", "plot_timeseries_interactive"])
    def test_jrc_id_geohash_kwarg_emits_deprecation_warning(self, jrc_test_dataset, method):
        """JRC static/interactive: legacy ``id_geohash=`` works but is deprecated."""
        ds = JRCDataset(jrc_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        with pytest.warns(DeprecationWarning, match="object_id"):
            fig = getattr(ds, method)(id_geohash=geohash)
        assert fig is not None
        if method == "plot_timeseries":
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        else:
            assert isinstance(fig, go.Figure)

    @pytest.mark.parametrize("method", ["plot_timeseries", "plot_timeseries_interactive"])
    def test_jrc_object_id_kwarg_no_warning(self, jrc_test_dataset, method):
        """JRC static/interactive: new ``object_id=`` works without warning."""
        ds = JRCDataset(jrc_test_dataset)
        geohash = ds.ds.coords["id_geohash"].values[0]

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)  # promote to error
            fig = getattr(ds, method)(object_id=geohash)
        assert fig is not None
        if method == "plot_timeseries":
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        else:
            assert isinstance(fig, go.Figure)
