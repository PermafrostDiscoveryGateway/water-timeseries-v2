import os
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader

# Set your EE project (or pass directly as ee_project parameter)
os.environ["EE_PROJECT"] = "pdg-project-406720"

# Create downloader instance
dl = EarthEngineDownloader(ee_auth=True, logger=logger)

ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    years=[2024],
    months=[7, 8],
    save_to_file="data5.zarr",  # Saves to downloads/data.zarr (relative path)
)

print('done downloading')