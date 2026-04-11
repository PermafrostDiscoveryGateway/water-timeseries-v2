import os
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader
import nest_asyncio
nest_asyncio.apply()

# Set your EE project (or pass directly as ee_project parameter)
os.environ["EE_PROJECT"] = "pdg-project-406720"

dl = EarthEngineDownloader(ee_auth=True, logger=logger)

ds = dl.download_dw_monthly(
    vector_dataset="/Users/helium/ncsa/pdg/water-timeseries-v2/tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    years=[2024],
    months=[7, 8],
    save_to_file="/Users/helium/ncsa/pdg/water-timeseries-v2/test_data.zarr",  # Saves to downloads/data.zarr (relative path)
)

print(f"We downloaded something")