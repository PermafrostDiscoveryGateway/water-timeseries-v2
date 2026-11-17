"""Tests for plotting functionality."""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from water_timeseries.breakpoint import SimpleBreakpoint
from water_timeseries.dataset import DWDataset, JRCDataset


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
        """Test that plot_timeseries creates a matplotlib figure."""
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


class TestTemporalResolutionDetection:
    """Hover date format must follow the dataset's temporal resolution."""

    def _dw(self, dates, attrs=None):
        import numpy as np
        import xarray as xr

        from water_timeseries.utils.data import dw_bandnames

        data = {var: (("id_geohash", "date"), np.full((1, len(dates)), 1.0)) for var in dw_bandnames}
        ds = xr.Dataset(data, coords={"id_geohash": ["lake"], "date": pd.to_datetime(dates)})
        if attrs:
            ds.attrs.update(attrs)
        return ds

    def test_declared_attribute_wins(self):
        """The downloader's attribute is trusted over the calendar."""
        from water_timeseries.utils.data import detect_temporal_resolution

        monthly = self._dw(["2020-06-01", "2020-07-01"], {"temporal_resolution": "monthly"})
        daily = self._dw(["2020-06-05", "2020-06-11"], {"temporal_resolution": "per_acquisition"})
        assert detect_temporal_resolution(monthly) == "monthly"
        assert detect_temporal_resolution(daily) == "per_acquisition"

    def test_falls_back_to_calendar_without_attribute(self):
        from water_timeseries.utils.data import detect_temporal_resolution

        assert detect_temporal_resolution(self._dw(["2020-06-01", "2020-07-01"])) == "monthly"
        assert detect_temporal_resolution(self._dw(["2020-06-05", "2020-06-11"])) == "per_acquisition"

    def test_attribute_overrides_a_daily_calendar(self):
        """A daily series that happens to land on month starts stays per-acquisition."""
        from water_timeseries.utils.data import detect_temporal_resolution

        ds = self._dw(["2020-06-01", "2020-07-01"], {"temporal_resolution": "per_acquisition"})
        assert detect_temporal_resolution(ds) == "per_acquisition"

    def test_plot_hover_uses_full_date_for_per_acquisition(self):
        ds = DWDataset(
            self._dw(
                ["2020-06-05", "2020-06-11", "2020-07-02"],
                {"temporal_resolution": "per_acquisition"},
            )
        )
        fig = ds.plot_timeseries_interactive("lake")
        assert fig.layout.xaxis.hoverformat == "%Y-%m-%d"

    def test_plot_hover_uses_month_for_monthly(self):
        ds = DWDataset(
            self._dw(["2020-06-01", "2020-07-01", "2020-08-01"], {"temporal_resolution": "monthly"})
        )
        fig = ds.plot_timeseries_interactive("lake")
        assert fig.layout.xaxis.hoverformat == "%Y-%m"

    def test_plotter_accepts_the_format_parameter(self):
        """The plotter itself must honour the new argument, not only the dataset wrapper."""
        import pandas as pd
        import plotly.graph_objects as go

        from water_timeseries.utils.plotting_dynamic import plot_water_time_series_dw_interactive

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-06-05", "2020-06-11"]),
                "variable": ["water", "water"],
                "value": [1.0, 2.0],
            }
        )
        fig = plot_water_time_series_dw_interactive(frame, hover_date_format="%d.%m.%Y")
        assert isinstance(fig, go.Figure)
        assert fig.layout.xaxis.hoverformat == "%d.%m.%Y"
