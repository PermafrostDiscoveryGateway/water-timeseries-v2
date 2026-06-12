import geopandas as gpd
from shapely.geometry import Point

df = gpd.read_parquet('downloads/data/Nitze_etal_Lakes_filtered_full_set_V2d.parquet')
point = Point(-121.0, 66.0) # Coordinates in the western half of Great Bear Lake

# Find lakes that intersect this point or are nearby
# For simplicity, filter by bounding box
western_half = df.cx[-123:-120, 65:67]
western_half = western_half[western_half.geometry.area > 0.1] # Just get large ones
western_half = western_half.sort_values('Area_start_ha', ascending=False)
for i, row in western_half.head(5).iterrows():
    print(f"ID: {row.get('id_geohash')}, Area: {row.get('Area_start_ha')}, Bounds: {row.geometry.bounds}")

