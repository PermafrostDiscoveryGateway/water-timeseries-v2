# Object identifiers (`id_field`)

Every dataset in this package identifies its objects (lakes, cells, sites) by one
coordinate/dimension. Its **name** is configurable; the **value** inside it is not a
name at all. Two words, used consistently:

| word | meaning | example |
|---|---|---|
| `id_field` | the *name* of the coordinate / column | `"id_geohash"`, `"lake_id"` |
| `object_id` | the *value* that selects one object | `"b7g4rf3n3x43"`, `12` |

`"id_geohash"` is the default everywhere, so existing code keeps working unchanged.

## Using a different identifier

```python
from water_timeseries.dataset import DWDataset

ds = xr.open_dataset("lake_exports.zarr")          # coordinate named "lake_id"
dataset = DWDataset(ds, id_field="lake_id")        # nothing else changes
dataset.object_ids_[:3]
dataset.plot_timeseries(object_id="lakeA")
```

Free functions take the name explicitly:

```python
from water_timeseries.utils.spatial import select_feature_by_id

feature = select_feature_by_id(lakes_gdf, "lakeA", id_field="lake_id")
```

`LakeDataset` records the name on the dataset (`ds.attrs["id_field"]`), so helpers that
receive only the xarray object can honour it:

```python
from water_timeseries.utils.data import resolve_id_field

resolve_id_field(dataset.ds)   # -> "lake_id"
```

Resolution order: `attrs["id_field"]` → the known names (`id_geohash` first) → the only
remaining non-`date` 1-D coordinate. When several candidates exist and none is declared,
it raises rather than guessing: selecting the wrong column silently is the bug this
design removes.

## Artefacts keep `id_geohash`

Files this package writes — break parquets, NRT parquets, PMTiles properties, dashboard
payloads — carry the identifier as **`id_geohash`**, whatever the source dataset called
it. That name is a wire contract for `pmtiles_build`, the MapLibre styles
(`promoteId`) and existing consumers. A custom `id_field` changes which column is *read*,
not which columns are *written*.

## Known limits

- The Streamlit dashboard (`run_map_viewer`) is geohash-only: it reads the contract
  columns and passes `id_column="id_geohash"`.
- `downloader` methods spell the same idea `name_attribute`; it already defaults to
  `"id_geohash"` and accepts any column name.
- `resolve_id_field` trusts `attrs` when present, and pandas/xarray do not guarantee
  `attrs` survive every transformation — pass `id_field` explicitly when it matters.
