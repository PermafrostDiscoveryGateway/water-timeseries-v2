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
        consolidated: bool = True,  # Add parameter with default True
) -> Path:
    """Save xarray dataset to file.

    Args:
        ds: The xarray dataset to save.
        save_path: Path to save the file. Format is determined by extension:
            - '.zarr' for Zarr format
            - '.nc' for NetCDF format
            If a relative path is provided and output_dir is specified,
            the file will be saved in that directory.
        output_dir: Directory for relative paths. If None and save_path is relative,
            the current working directory is used.
        logger: Logger for logging progress. If None, print statements are used.
        consolidated: For Zarr format, whether to use consolidated metadata.
            Default True (recommended for better compatibility).

    Returns:
        Path: The resolved path where the dataset was saved.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(save_path)

    # Handle relative path
    if not path.is_absolute() and output_dir is not None:
        path = Path(output_dir) / path

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine format from extension
    ext = path.suffix.lower()

    # Logging helper
    def _log(msg: str):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    _log(f"Saving to {ext[1:].upper()} format: {path}")

    if ext == ".zarr":
        # Handle string coordinates properly
        encoding = {}
        for coord in ds.coords:
            if ds[coord].dtype.kind in ['U', 'S']:  # String type
                # Convert string coordinates to object dtype for Zarr
                encoding[coord] = {'dtype': 'object'}
                _log(f"  Converting coordinate '{coord}' to object dtype for Zarr")

        # Save with consolidated metadata (creates .zgroup!)
        ds.to_zarr(
            path,
            mode="w",
            consolidated=consolidated,
            encoding=encoding if encoding else None
        )

        if consolidated:
            _log(f"  Saved with consolidated metadata (creates .zgroup file)")
        else:
            _log(f"  Saved without consolidated metadata")

    elif ext == ".nc":
        # For NetCDF, handle string coordinates automatically
        ds.to_netcdf(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use '.zarr' or '.nc'.")

    _log(f"Dataset saved successfully to {path}")

    # Verify Zarr save was successful
    if ext == ".zarr" and consolidated:
        zgroup_path = path / '.zgroup'
        if zgroup_path.exists():
            _log(f"  ✓ Verified .zgroup file created")
        else:
            _log(f"  ⚠️ Warning: .zgroup file not found (may cause reading issues)")

    return path


def load_xarray_dataset(
    path: Union[str, Path],
    format: Optional[str] = None,
) -> xr.Dataset:
    """Load xarray dataset from file.

    Args:
        path: Path to the dataset file.
        format: Format of the file ('zarr' or 'netcdf'). If None, auto-detected
            from extension.

    Returns:
        xr.Dataset: The loaded dataset.

    Raises:
        ValueError: If the file format is not supported.
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
        return xr.open_zarr(path)
    elif format == "netcdf":
        return xr.open_dataset(path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'zarr' or 'netcdf'.")
