# API Reference

This section provides comprehensive API documentation for the water-timeseries package.

## Modules

### Core Dataset Classes
- [Dataset](dataset.md) - Main dataset processing classes (`LakeDataset`, `DWDataset`, `JRCDataset`)

### Breakpoint Detection
- [Breakpoint](breakpoint.md) - Breakpoint detection methods for identifying changes in water extent

### Utilities
- [Utils](utils.md) - Plotting and helper utilities

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
