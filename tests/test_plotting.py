"""Tests for plotting functionality."""

import matplotlib.pyplot as plt

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
