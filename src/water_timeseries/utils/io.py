"""Input/Output utilities for water timeseries data."""

from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import xarray as xr
from loguru import logger as logger


def is_remote_path(path: Union[str, Path]) -> bool:
    """Check if path is a remote URL/URI (e.g. gs://, s3://, http://, https://)."""
    if isinstance(path, Path):
        return False
    path_str = str(path)
    return any(path_str.startswith(proto) for proto in ("gs://", "s3://", "http://", "https://"))


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
    file_path_str = str(file_path)
    is_remote = is_remote_path(file_path_str)

    if is_remote:
        # Determine suffix from remote string path
        clean_path = file_path_str.split("?")[0].split("#")[0]
        suffix = "." + clean_path.split(".")[-1].lower() if "." in clean_path else ""
    else:
        file_path = Path(file_path)
        if not file_path.exists():
            if logger:
                logger.warning(f"Vector dataset file not found: {file_path}")
            raise FileNotFoundError(f"Vector dataset file not found: {file_path}")
        suffix = file_path.suffix.lower()

    if logger:
        logger.info(f"Loading vector dataset from {file_path_str}")

    # GeoPackage, Shapefile, GeoJSON formats
    if suffix in [".gpkg", ".shp", ".geojson", ".gjson"]:
        vector_ds = gpd.read_file(file_path_str if is_remote else file_path)
    elif suffix in [".parquet"]:
        vector_ds = gpd.read_parquet(file_path_str if is_remote else file_path)
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

    Returns:
        Path: The resolved path where the dataset was saved.

    Raises:
        ValueError: If the file extension is not supported.
    """

    # Logging helper
    def _log(msg: str):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

        # Logging helper

    def _log_warning(msg: str):
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg)

    if output_dir is not None:
        _log_warning("Setting an output_dir is being ignored and will be removed eventually!")

    path = Path(save_path).absolute()

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine format from extension
    ext = path.suffix.lower()

    _log(f"Saving to {ext[1:].upper()} format: {path}")

    if ext == ".zarr":
        ds.to_zarr(path, mode="w")
    elif ext == ".nc":
        ds.to_netcdf(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use '.zarr' or '.nc'.")

    _log(f"Dataset saved successfully to {path}")

    return path


def load_xarray_dataset(
    path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs,
) -> xr.Dataset:
    """Load xarray dataset from file.

    Args:
        path: Path to the dataset file.
        format: Format of the file ('zarr' or 'netcdf'). If None, auto-detected
            from extension.
        **kwargs: Additional arguments passed to xr.open_zarr or xr.open_dataset.

    Returns:
        xr.Dataset: The loaded dataset.

    Raises:
        ValueError: If the file format is not supported.
    """
    path_str = str(path)
    is_remote = is_remote_path(path_str)

    if format is None:
        if is_remote:
            clean_path = path_str.split("?")[0].split("#")[0]
            ext = "." + clean_path.split(".")[-1].lower() if "." in clean_path else ""
            if clean_path.endswith(".zarr/"):
                ext = ".zarr"
        else:
            ext = Path(path).suffix.lower()

        if ext == ".zarr":
            format = "zarr"
        elif ext in (".nc", ".ncin"):
            format = "netcdf"
        else:
            raise ValueError(f"Cannot auto-detect format for extension: {ext}")

    target_path = path_str if is_remote else Path(path)

    if format == "zarr":
        return xr.open_zarr(target_path, **kwargs)
    elif format == "netcdf":
        if is_remote:
            import fsspec
            f = fsspec.open(target_path)
            return xr.open_dataset(f.open(), **kwargs)
        else:
            return xr.open_dataset(target_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'zarr' or 'netcdf'.")
