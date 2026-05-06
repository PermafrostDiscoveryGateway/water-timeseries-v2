import xarray as xr
import netCDF4 as nc
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union, Optional, List, Tuple, Dict, Any


def diagnose_netcdf(nc_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Thoroughly diagnose a NetCDF file and extract all time-related information.

    Parameters:
    -----------
    nc_path : str or Path
        Path to the NetCDF file

    Returns:
    --------
    dict : Dictionary containing diagnostic information
    """
    nc_path = Path(nc_path)

    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    file_size = nc_path.stat().st_size / (1024 * 1024)  # MB
    print(f"\n{'=' * 70}")
    print(f"NetCDF Diagnostic: {nc_path.name}")
    print(f"{'=' * 70}")
    print(f"File size: {file_size:.2f} MB")

    diagnostic = {
        'path': str(nc_path),
        'size_mb': file_size,
        'variables': [],
        'dimensions': {},
        'time_info': None,
        'has_time': False
    }

    # Try both xarray and netCDF4 approaches
    try:
        # Method 1: Using xarray (recommended for most cases)
        print("\n📊 Opening with xarray...")
        ds = xr.open_dataset(nc_path)

        print(f"\nDimensions:")
        for dim, size in ds.dims.items():
            print(f"  {dim}: {size}")
            diagnostic['dimensions'][dim] = size

        print(f"\nCoordinates:")
        for coord in ds.coords:
            print(f"  {coord}: {ds[coord].dtype} - {ds[coord].values[:3] if len(ds[coord]) > 0 else 'empty'}...")

        print(f"\nData Variables:")
        for var in ds.data_vars:
            print(f"  {var}: {ds[var].dims} - {ds[var].dtype}")
            diagnostic['variables'].append(var)

        # Look for time
        time_info = extract_time_from_xarray(ds)
        if time_info:
            diagnostic['has_time'] = True
            diagnostic['time_info'] = time_info
            print(f"\n✅ TIME DIMENSION FOUND!")
            print(f"  Name: {time_info['name']}")
            print(f"  Shape: {time_info['shape']}")
            print(f"  Range: {time_info['min']} to {time_info['max']}")
            print(f"  Number of timesteps: {time_info['n_timesteps']}")

            if time_info['is_regular']:
                print(f"  Regular interval: {time_info['frequency']}")
            else:
                print(f"  ⚠️ Irregular intervals detected")

        else:
            print(f"\n❌ No time dimension found in xarray")

            # Try to find time-like variables
            print("\nChecking for time-like variables in data_vars:")
            for var in ds.data_vars:
                if any(t in var.lower() for t in ['time', 'date', 'datetime']):
                    print(f"  Possible time variable: {var}")
                    try:
                        values = ds[var].values
                        print(f"    First 5 values: {values[:5]}")
                    except:
                        print(f"    Could not read values")

        ds.close()

    except Exception as e:
        print(f"❌ xarray failed to open: {e}")
        print("\nTrying with netCDF4 directly...")

        # Method 2: Using netCDF4 (more low-level)
        try:
            ds_nc = nc.Dataset(nc_path, 'r')

            print(f"\nNetCDF4 Info:")
            print(f"  Format: {ds_nc.file_format}")
            print(f"  Dimensions: {list(ds_nc.dimensions.keys())}")
            for dim, size in ds_nc.dimensions.items():
                print(f"    {dim}: {len(size) if hasattr(size, '__len__') else size}")

            print(f"  Variables: {list(ds_nc.variables.keys())}")

            # Check for time
            for var_name in ds_nc.variables:
                if any(t in var_name.lower() for t in ['time', 'date', 'datetime']):
                    var = ds_nc.variables[var_name]
                    print(f"\n  Found time variable: {var_name}")
                    print(f"    Dimensions: {var.dimensions}")
                    print(f"    Shape: {var.shape}")
                    print(f"    Units: {var.units if hasattr(var, 'units') else 'Not specified'}")

                    try:
                        # Try to read a few values
                        values = var[:5] if len(var) > 0 else []
                        print(f"    First 5 values: {values}")
                    except:
                        print(f"    Could not read values")

            ds_nc.close()

        except Exception as e2:
            print(f"❌ netCDF4 also failed: {e2}")

    print(f"\n{'=' * 70}")
    return diagnostic


def extract_time_from_xarray(ds: xr.Dataset) -> Optional[Dict]:
    """
    Extract time information from an xarray dataset.
    """
    # Common time coordinate names
    time_candidates = ['time', 'Time', 't', 'datetime', 'date',
                       'valid_time', 'forecast_time', 'reference_time']

    time_coord = None

    # Check coordinates first
    for candidate in time_candidates:
        if candidate in ds.coords:
            time_coord = ds[candidate]
            time_name = candidate
            break

    # If not in coords, check data_vars
    if time_coord is None:
        for candidate in time_candidates:
            if candidate in ds.data_vars:
                time_coord = ds[candidate]
                time_name = candidate
                break

    if time_coord is None:
        return None

    # Extract time values
    try:
        time_values = time_coord.values

        # Try to convert to pandas datetime
        try:
            pd_times = pd.to_datetime(time_values)
            min_time = pd_times.min()
            max_time = pd_times.max()
            is_regular = check_regular_interval(pd_times)
            frequency = infer_frequency(pd_times) if is_regular else None

            return {
                'name': time_name,
                'shape': time_coord.shape,
                'dtype': str(time_coord.dtype),
                'values': pd_times,
                'min': min_time,
                'max': max_time,
                'n_timesteps': len(pd_times),
                'is_regular': is_regular,
                'frequency': frequency,
                'units': time_coord.attrs.get('units', None),
                'calendar': time_coord.attrs.get('calendar', None)
            }
        except:
            # Return raw values if can't convert
            return {
                'name': time_name,
                'shape': time_coord.shape,
                'dtype': str(time_coord.dtype),
                'values': time_values,
                'min': time_values.min() if len(time_values) > 0 else None,
                'max': time_values.max() if len(time_values) > 0 else None,
                'n_timesteps': len(time_values),
                'is_regular': False,
                'frequency': None,
                'units': time_coord.attrs.get('units', None),
                'calendar': time_coord.attrs.get('calendar', None)
            }
    except Exception as e:
        print(f"Error extracting time values: {e}")
        return None


def check_regular_interval(times: pd.DatetimeIndex, tolerance='1D') -> bool:
    """Check if time intervals are regular."""
    if len(times) < 2:
        return True

    diffs = times[1:] - times[:-1]
    mode_diff = diffs.mode()

    if len(mode_diff) == 0:
        return False

    # Check if all diffs are within tolerance of the mode
    tolerance_td = pd.Timedelta(tolerance)
    return all(abs(diff - mode_diff[0]) <= tolerance_td for diff in diffs)


def infer_frequency(times: pd.DatetimeIndex) -> str:
    """Infer the frequency of a regular time series."""
    if len(times) < 2:
        return 'unknown'

    diff = times[1] - times[0]

    # Map timedeltas to frequency strings
    freq_map = {
        pd.Timedelta(days=1): 'D',
        pd.Timedelta(hours=1): 'H',
        pd.Timedelta(minutes=1): 'T',
        pd.Timedelta(seconds=1): 'S',
        pd.Timedelta(days=7): 'W',
        pd.Timedelta(days=30): 'M',
        pd.Timedelta(days=365): 'Y'
    }

    return freq_map.get(diff, 'unknown')


def get_netcdf_dates(
        nc_path: Union[str, Path],
        time_coord_name: Optional[str] = None,
        as_pandas: bool = True
) -> np.ndarray:
    """
    Extract dates from a NetCDF file.

    Parameters:
    -----------
    nc_path : str or Path
        Path to NetCDF file
    time_coord_name : str, optional
        Name of time coordinate (auto-detected if None)
    as_pandas : bool, default=True
        Return pandas DatetimeIndex if True

    Returns:
    --------
    np.ndarray : Array of dates/times
    """
    nc_path = Path(nc_path)

    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    # Open with xarray
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        raise ValueError(f"Failed to open NetCDF file: {e}")

    # Find time coordinate
    if time_coord_name is None:
        time_candidates = ['time', 'Time', 't', 'datetime', 'date',
                           'valid_time', 'forecast_time', 'reference_time']

        # Check coordinates first
        for candidate in time_candidates:
            if candidate in ds.coords:
                time_coord_name = candidate
                break

        # If not found, check data variables
        if time_coord_name is None:
            for candidate in time_candidates:
                if candidate in ds.data_vars:
                    time_coord_name = candidate
                    break

        # Look for any variable with time units
        if time_coord_name is None:
            for var_name, var in ds.variables.items():
                if 'units' in var.attrs and 'since' in var.attrs['units'].lower():
                    time_coord_name = var_name
                    break

    if time_coord_name is None:
        ds.close()
        raise ValueError(
            f"No time dimension found in {nc_path}. "
            f"Available coordinates: {list(ds.coords.keys())}"
        )

    # Extract time values
    time_var = ds[time_coord_name]
    time_values = time_var.values

    if as_pandas:
        try:
            # Try direct conversion
            dates = pd.to_datetime(time_values)
        except:
            # Try using netCDF4 for more robust conversion
            try:
                import netCDF4
                with nc.Dataset(nc_path, 'r') as nc_ds:
                    time_var_nc = nc_ds.variables[time_coord_name]

                    # Get units and calendar
                    units = getattr(time_var_nc, 'units', None)
                    calendar = getattr(time_var_nc, 'calendar', 'standard')

                    if units and 'since' in units:
                        # Use netCDF4's num2date
                        dates = netCDF4.num2date(time_values, units, calendar)
                        dates = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
                    else:
                        dates = time_values
            except:
                dates = time_values
    else:
        dates = time_values

    ds.close()
    return dates


def compare_zarr_and_netcdf(zarr_path: Path, netcdf_path: Path):
    """
    Compare a Zarr dataset with a NetCDF file of the same data.
    Useful for debugging download issues.
    """
    print(f"\n{'=' * 70}")
    print("Comparing Zarr vs NetCDF")
    print(f"{'=' * 70}")

    # Check Zarr
    print("\n🔍 ZARR DATASET:")
    zarr_valid = (zarr_path / '.zgroup').exists()
    print(f"  Valid: {zarr_valid}")
    if zarr_valid:
        try:
            import zarr
            root = zarr.open_group(zarr_path, mode='r')
            arrays = list(root.array_keys())
            print(f"  Arrays: {len(arrays)}")
            size = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
            print(f"  Size: {size / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"  Error reading: {e}")
    else:
        print("  ❌ Not a valid Zarr dataset (missing .zgroup)")

    # Check NetCDF
    print("\n🔍 NETCDF FILE:")
    if netcdf_path.exists():
        size = netcdf_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {size:.2f} MB")

        try:
            dates = get_netcdf_dates(netcdf_path, as_pandas=True)
            print(f"  ✅ Valid NetCDF file")
            print(f"  Number of timesteps: {len(dates)}")
            print(f"  Date range: {dates.min()} to {dates.max()}")
        except Exception as e:
            print(f"  ❌ Error reading: {e}")
    else:
        print(f"  ❌ File not found: {netcdf_path}")


# Example usage
if __name__ == "__main__":
    import os

    os.environ['ZARR_V3_EXPERIMENTAL_API'] = '0'
    os.environ['XARRAY_BACKEND_ZARR_USE_SYNCHRONOUS'] = '1'

    import xarray as xr

    # This should now work even in debug mode
    ds = xr.open_zarr("downloads/data5.zarr", consolidated=False)
    print(f"Success! Loaded {len(ds.date)} dates")
    import zarr
    import xarray as xr

    zarr_path = "downloads/data5.zarr"

    # Method that worked (zarr-python directly)
    root = zarr.open_group(zarr_path, mode='r')
    print("Arrays found:", list(root.array_keys()))

    # Try to reconstruct as xarray
    # This is a workaround to read the data
    data_dict = {}
    for array_name in root.array_keys():
        data_dict[array_name] = root[array_name][:]

    print(f"Successfully read {len(data_dict)} arrays")

    import zarr
    import xarray as xr

    # test_read_zarr.py
    import xarray as xr
    from pathlib import Path

    # Try to read the Zarr file you just created
    zarr_path = Path("downloads/data5.zarr")

    print(f"Checking Zarr at: {zarr_path}")
    print(f"Directory exists: {zarr_path.exists()}")

    try:
        # Try to open with xarray (this is the real test)
        ds = xr.open_zarr(zarr_path, consolidated=False)  # Note: consolidated=False for v3

        print(f"\n✅ SUCCESS! Zarr file loaded correctly")
        print(f"Dimensions: {dict(ds.dims)}")
        print(f"Coordinates: {list(ds.coords.keys())}")
        print(f"Data variables: {list(ds.data_vars.keys())}")

        # Check the dates
        if 'date' in ds.coords:
            print(f"\nDate coordinate:")
            print(f"  Type: {ds['date'].dtype}")
            print(f"  Values: {ds['date'].values[:3]}...")

        # Check a small sample of data
        if 'water' in ds.data_vars:
            print(f"\nWater data sample:")
            print(f"  Shape: {ds['water'].shape}")
            print(f"  First few values: {ds['water'].values[0, :3]}")

    except Exception as e:
        print(f"\n❌ FAILED to read Zarr: {e}")

        # Try alternative read method
        print("\nTrying alternative read method...")
        try:
            import zarr

            root = zarr.open_group(zarr_path, mode='r')
            print(f"Zarr groups: {list(root.group_keys())}")
            print(f"Zarr arrays: {list(root.array_keys())}")
        except Exception as e2:
            print(f"Alternative also failed: {e2}")

    print(f"Zarr version: {zarr.__version__}")
    print(f"Xarray version: {xr.__version__}")

    # Check if consolidated metadata is supported
    print(f"Consolidated metadata support: {hasattr(zarr, 'consolidate_metadata')}")
    # Example 1: Diagnose a NetCDF file
    # zarr_dataset = "/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/utils/2023_6_9_2024_dw_download.zarr"
    zarr_dataset = "/Users/helium/ncsa/pdg/water-timeseries-v2/downloads/data5.zarr"
    netcdf_file = "/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/utils/2023_6_9_2024_dw_download.nc"  # Change this
    diagnostic = diagnose_netcdf(netcdf_file)

    # Example 2: Extract dates
    if diagnostic['has_time']:
        dates = get_netcdf_dates(netcdf_file, as_pandas=True)
        print(f"\nExtracted {len(dates)} dates:")
        print(f"  First 5: {dates[:5]}")
        print(f"  Last 5: {dates[-5:]}")

    # Example 3: Compare Zarr vs NetCDF (for debugging)
    # zarr_dataset = Path("path/to/data.zarr")
    netcdf_file_path = Path(netcdf_file)
    compare_zarr_and_netcdf(Path(zarr_dataset), netcdf_file_path)