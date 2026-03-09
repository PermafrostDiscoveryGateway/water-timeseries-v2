# water-timeseries-v2

Automated analysis of water timeseries data from satellite imagery and remote sensing sources.

## Documentation

**📖 Full Documentation**: [View Documentation](https://PermafrostDiscoveryGateway.github.io/water-timeseries-v2/)

The documentation includes:
- Getting started guide
- API reference (auto-generated from code)
- Usage examples
- Tutorial notebooks

Documentation is automatically built and deployed on every push to `main` using GitHub Actions.

## Features

- **Dynamic World Handler**: Process Dynamic World land cover classifications
- **JRC Water Handler**: Handle JRC water occurrence and classification data
- **Data Normalization**: Automatic normalization and scaling of time series
- **Breakpoint Detection**: Statistical (SimpleBreakpoint) and advanced (RBEAST) methods for detecting water extent changes
- **Batch Processing**: Efficient processing of multiple spatial entities
- **Comprehensive Testing**: Full test coverage including breakpoint detection, normalization, and integration tests

## Quick Start

```python
from water_timeseries.dataset import DWDataset
import xarray as xr

# Load data
ds = xr.open_dataset("water_data.nc")

# Process with Dynamic World handler
processor = DWDataset(ds)

# Access time series
water_extent = processor.ds_normalized[processor.water_column]

# Access normalized time series
water_extent = processor.ds_normalized["water"]
```

## Installation

```bash
git clone https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2
cd water-timeseries-v2
```

`pip install .` or `uv sync`

Or for development:

```bash
git clone https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2
cd water-timeseries-v2
pip install -e ".[dev]"
```

## Main Classes

### Datasets

- **DWDataset**: Dynamic World land cover processor
- **JRCDataset**: JRC water classification processor

### Breakpoints

- **SimpleBreakpoint**: Statistical breakpoint detection
- **BeastBreakpoint**: Advanced RBEAST-based detection

## Testing

The package includes comprehensive tests covering:
- Dataset normalization and masking
- Breakpoint detection methods (Simple and RBEAST)
- Batch processing functionality
- Integration tests with real and synthetic data

Run tests with: `pytest`

## Contributing

We welcome contributions! Please ensure you:
1. Add docstrings to new functions and classes (Google style)
2. Update documentation in the `docs/` folder
3. Run tests before submitting PRs

## License

[Add your license here]

## Author

Ingmar Nitze
