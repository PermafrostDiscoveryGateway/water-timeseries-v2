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

## Artefacts keep `id_geohash`, and carry the source name beside it

Files this package writes — break parquets, NRT parquets, PMTiles properties, dashboard
payloads — carry the identifier as **`id_geohash`**, whatever the source dataset called
it. That name is a wire contract for `pmtiles_build`, the MapLibre styles
(`promoteId`) and existing consumers.

When the source name differs, the artefact also **keeps** it, so a reader can map back
to the dataset:

```python
from water_timeseries.utils.data import ensure_id_columns

df = ensure_id_columns(df, "lake_id")      # columns: lake_id, id_geohash, ...
```

`ensure_id_columns` is additive by construction — it never renames, drops or reorders.
With the default `id_field="id_geohash"` it is a no-op, so existing files and their
columns are unchanged. `calculate_break` (simple / beast / NRT) already returns such a
frame: the index keeps the dataset's own name and `id_geohash` is present as a column.

A new column is only ever added when the names differ. Consumers that assert an exact
column list will see one extra column in that case — the default path emits nothing new.

## Download → break pipeline

The downloaders name the identifier coordinate after the vector column they were given
(`name_attribute`), and now record it on the dataset:

```python
dl = EarthEngineDownloader(ee_auth=True, logger=logger)
ds = dl.download_dw_monthly(vector_dataset="lakes.parquet", name_attribute="lake_id")
ds.attrs["id_field"]        # "lake_id"
```

From there the break pipeline resolves the name itself — no flag needed:

```python
pipeline = BreakpointPipeline(
    water_dataset_file="lakes_lake_id.nc",
    output_file="breaks.parquet",
    break_method="simple",
)
pipeline.id_field   # "lake_id"  (from the dataset's attrs)
```

or pass it explicitly, which also covers a dataset that records nothing:

```bash
water-timeseries breakpoint-analysis-historical \
    --water-dataset-file lakes_lake_id.nc \
    --output-file breaks.parquet --id-field lake_id
```

The output parquet then carries **both** names: `id_geohash` (the contract every
downstream consumer reads) plus `lake_id` (the source name, as the index).

## Known limits

- The Streamlit dashboard (`run_map_viewer`) is geohash-only: it reads the contract
  columns and passes `id_column="id_geohash"`.
- `downloader` methods spell the same idea `name_attribute`; it already defaults to
  `"id_geohash"` and accepts any column name.
- `resolve_id_field` trusts `attrs` when present, and pandas/xarray do not guarantee
  `attrs` survive every transformation — pass `id_field` explicitly when it matters.
