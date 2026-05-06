"""Input/Output utilities for water timeseries data."""

from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import xarray as xr
from loguru import logger as logger


def load_vector_dataset(
    file_path: Union[str, Path],
    logger: Optional[logger] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Load a vector dataset from file based on file extension.

    Supports GeoPackage, Shapefile, GeoJSON, and Parquet formats.

    Args:
        file_path: Path to the vector dataset file.
        logger: Optional logger instance for logging messages.

    Returns:
        GeoDataFrame if successful, None otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        if logger:
            logger.warning(f"Vector dataset file not found: {file_path}")
        raise FileNotFoundError(f"Vector dataset file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if logger:
        logger.info(f"Loading vector dataset from {file_path}")

    # GeoPackage, Shapefile, GeoJSON formats
    if suffix in [".gpkg", ".shp", ".geojson", ".gjson"]:
        vector_ds = gpd.read_file(file_path)
    elif suffix in [".parquet"]:
        vector_ds = gpd.read_parquet(file_path)
    else:
        if logger:
            logger.warning(f"Unsupported vector file format: {suffix}")
        return None

    return vector_ds


def save_xarray_dataset(
        ds: xr.Dataset,
        save_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        logger=None,
        consolidated: bool = True,
) -> Path:
    """Save xarray dataset to file with Zarr v3 compatibility."""
    path = Path(save_path)

    # Handle relative path
    if not path.is_absolute() and output_dir is not None:
        path = Path(output_dir) / path

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine format from extension
    ext = path.suffix.lower()

    # Logging helper
    def _log(msg: str, level="INFO"):
        if logger is not None:
            if level == "ERROR":
                logger.error(msg)
            elif level == "WARNING":
                logger.warning(msg)
            else:
                logger.info(msg)
        else:
            print(f"[{level}] {msg}")

    _log(f"Saving to {ext[1:].upper()} format: {path}")

    if ext == ".zarr":
        # Create a copy to work with
        ds_to_save = ds.copy()

        # Convert dates to datetime64 for better compatibility
        for coord_name in list(ds_to_save.coords.keys()):
            if coord_name == 'date' or 'date' in coord_name.lower():
                coord = ds_to_save.coords[coord_name]
                if coord.dtype.kind in ['U', 'S', 'O']:
                    try:
                        import pandas as pd
                        _log(f"  Converting '{coord_name}' to datetime64...")
                        datetime_values = pd.to_datetime(coord.values)
                        ds_to_save.coords[coord_name] = xr.DataArray(
                            datetime_values,
                            dims=coord.dims,
                            attrs=coord.attrs
                        )
                        _log(f"    ✓ Converted to datetime64")
                    except Exception as e:
                        _log(f"    ✗ Conversion failed: {e}", "WARNING")

        # For Zarr v3, we need to use a different approach
        # Option 1: Use consolidated=False (recommended for Zarr v3)
        try:
            _log("Attempting Zarr save with consolidated=False (Zarr v3 compatible)...")

            # Prepare encoding for string coordinates
            encoding = {}
            for coord_name in ds_to_save.coords:
                coord = ds_to_save.coords[coord_name]
                # For string/object coordinates, use object dtype with no compressor
                if coord.dtype.kind in ['U', 'S', 'O']:
                    encoding[coord_name] = {
                        'dtype': 'object',
                        'compressor': None,
                        'filters': None
                    }
                    _log(f"  Encoding '{coord_name}' as object dtype")

            # Save with consolidated=False (works better with Zarr v3)
            ds_to_save.to_zarr(
                path,
                mode="w",
                consolidated=False,  # Don't use consolidated metadata for v3
                encoding=encoding if encoding else None
            )

            # Verify the save
            if path.exists():
                # Check for any .zarray files (v3 uses different structure)
                zarray_files = list(path.rglob('*.zarray'))
                zmetadata_files = list(path.rglob('*.zmetadata'))

                if zarray_files or zmetadata_files:
                    _log(
                        f"✓ Zarr save successful! Found {len(zarray_files)} .zarray files, {len(zmetadata_files)} .zmetadata files")
                    return path
                else:
                    _log(f"⚠️ Save completed but no array files found", "WARNING")
                    # Try to read it back to verify
                    try:
                        test_read = xr.open_zarr(path)
                        _log(f"✓ Successfully verified Zarr can be read back")
                        return path
                    except Exception as read_err:
                        _log(f"✗ Cannot read back Zarr: {read_err}", "ERROR")
                        raise

        except Exception as e:
            _log(f"First save attempt failed: {e}", "WARNING")

            # Option 2: Use Zarr v2 compatibility mode
            try:
                _log("Attempting Zarr save with v2 compatibility...")

                # Force Zarr v2 format by using a different storage option
                # This is more reliable for string/object data
                import zarr

                # Create Zarr group with v2 format
                store = zarr.DirectoryStore(path)
                root = zarr.group(store, overwrite=True, zarr_version=2)  # Force v2

                # Use xarray to save to the v2 group
                ds_to_save.to_zarr(
                    store,
                    mode="w",
                    consolidated=consolidated,
                    group=None
                )

                _log(f"✓ Zarr save successful with v2 format!")

                # Verify
                zgroup_path = path / '.zgroup'
                if zgroup_path.exists():
                    _log(f"  Found .zgroup file (v2 format)")
                else:
                    _log(f"  Note: No .zgroup file (v3 format)")

                return path

            except Exception as e2:
                _log(f"All save attempts failed: {e2}", "ERROR")
                raise

    elif ext == ".nc":
        _log("Saving as NetCDF...")
        ds.to_netcdf(path)
        _log(f"✓ NetCDF saved successfully")

    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use '.zarr' or '.nc'.")

    return path

def load_xarray_dataset(
        path: Union[str, Path],
        format: Optional[str] = None,
        decode_times: bool = True,
) -> xr.Dataset:
    """Load xarray dataset from file with proper handling of Zarr files.

    Args:
        path: Path to the dataset file.
        format: Format of the file ('zarr' or 'netcdf'). If None, auto-detected
            from extension.
        decode_times: For NetCDF files, whether to decode time units (default: True).

    Returns:
        xr.Dataset: The loaded dataset.
    """
    path = Path(path)

    if format is None:
        ext = path.suffix.lower()
        if ext == ".zarr":
            format = "zarr"
        elif ext == ".nc":
            format = "netcdf"
        else:
            raise ValueError(f"Cannot auto-detect format for extension: {ext}")

    if format == "zarr":
        # For Zarr, we might need to handle object-typed coordinates
        ds = xr.open_zarr(path)

        # Try to convert object-typed date coordinates back to datetime
        for coord_name in ds.coords:
            if coord_name == 'date' or 'date' in coord_name.lower():
                if ds[coord_name].dtype == object:
                    try:
                        import pandas as pd
                        # Convert object array to datetime
                        ds[coord_name] = xr.DataArray(
                            pd.to_datetime(ds[coord_name].values),
                            dims=ds[coord_name].dims,
                            attrs=ds[coord_name].attrs
                        )
                    except Exception:
                        pass  # Keep as object if conversion fails

        return ds

    elif format == "netcdf":
        return xr.open_dataset(path, decode_times=decode_times)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'zarr' or 'netcdf'.")