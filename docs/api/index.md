# API Reference

This section provides comprehensive API documentation for the water-timeseries package.

## Modules

### Core Dataset Classes
- [Dataset](dataset.md) - Main dataset processing classes (`LakeDataset`, `DWDataset`, `JRCDataset`)

### Breakpoint Detection
- [Breakpoint](breakpoint.md) - Breakpoint detection methods for identifying changes in water extent

### Data Download
- [Downloader](downloader.md) - Google Earth Engine data downloader (`EarthEngineDownloader`)

### Utilities
- [Utils](utils.md) - Data handling, spatial helpers, Earth Engine and helper utilities

### Guides
- [Object Identifiers](../id_field.md) - Using a custom `id_field` instead of `id_geohash`
- [Sentinel-2 Imagery](../sentinel2_imagery.md) - Fetching an S2 cube and rendering RGB / CIR composites

## Usage

All modules are designed to work with xarray datasets:

```python
import xarray as xr
from water_timeseries.dataset import DWDataset

# Load data
ds = xr.open_dataset("data.nc")

# Process
processor = DWDataset(ds)
```
