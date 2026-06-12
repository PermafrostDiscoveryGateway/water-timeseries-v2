import geopandas as gpd

df = gpd.read_parquet('downloads/data/Nitze_etal_Lakes_filtered_full_set_V2d.parquet')
lake1 = df[df['id_geohash'] == 'c7erdc6d3f8h']
lake2 = df[df['id_geohash'] == 'c7fu1xfk5bnp']

print(f"c7erdc6d3f8h area: {lake1['Area_start_ha'].iloc[0] if len(lake1) else 0}")
print(f"c7fu1xfk5bnp area: {lake2['Area_start_ha'].iloc[0] if len(lake2) else 0}")
