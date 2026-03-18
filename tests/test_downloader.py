"""
Test suite for EarthEngineDownloader class.

This module contains comprehensive tests for authentication, downloading
Dynamic World data, JRC data, and error handling.
"""

import os
import unittest.mock as mock
from pathlib import Path

import pytest

from water_timeseries.downloader import EarthEngineDownloader


class TestEarthEngineDownloaderAuthentication:
    """Test authentication and initialization of EarthEngineDownloader."""

    def test_init_with_valid_ee_project(self):
        """Test initialization with a valid ee_project argument."""
        downloader = EarthEngineDownloader(ee_project="my-valid-project")
        assert downloader.ee_project == "my-valid-project"

    def test_init_with_empty_string_ee_project(self):
        """Test that empty string ee_project raises ValueError."""
        with pytest.raises(ValueError, match="ee_project must be provided or set as EE_PROJECT environment variable"):
            EarthEngineDownloader(ee_project="")

    def test_init_with_none_and_env_var_set(self, monkeypatch):
        """Test initialization with None ee_project uses EE_PROJECT env var."""
        monkeypatch.setenv("EE_PROJECT", "env-project-id")
        downloader = EarthEngineDownloader(ee_project=None)
        assert downloader.ee_project == "env-project-id"

    def test_init_with_none_and_no_env_var(self, monkeypatch):
        """Test that None ee_project without env var raises ValueError."""
        monkeypatch.delenv("EE_PROJECT", raising=False)
        with pytest.raises(ValueError, match="ee_project must be provided"):
            EarthEngineDownloader(ee_project=None)

    def test_argument_overwrites_env_var(self, monkeypatch):
        """Test that ee_project argument takes precedence over EE_PROJECT env var."""
        monkeypatch.setenv("EE_PROJECT", "env-project-id")
        downloader = EarthEngineDownloader(ee_project="arg-project-id")
        assert downloader.ee_project == "arg-project-id"

    def test_init_with_none_uses_env_var_even_with_auth(self, monkeypatch):
        """Test that None ee_project works with ee_auth=True when env var is set."""
        monkeypatch.setenv("EE_PROJECT", "env-project-id")
        with mock.patch("geemap.ee_initialize"):
            downloader = EarthEngineDownloader(ee_project=None, ee_auth=True)
            assert downloader.ee_project == "env-project-id"
            assert downloader.ee_auth is True

    def test_invalid_env_var_type(self, monkeypatch):
        """Test that non-string EE_PROJECT env var is ignored."""
        monkeypatch.setenv("EE_PROJECT", "")
        with pytest.raises(ValueError, match="ee_project must be provided"):
            EarthEngineDownloader(ee_project=None)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
