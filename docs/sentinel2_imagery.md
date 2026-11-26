# Sentinel-2 imagery: RGB and CIR composites

Two functions turn a lake polygon into a picture of the satellite imagery behind its
water series. Useful for checking whether a drop is real drainage or an ice/cloud
artefact, and for showing shoreline change without plotting the numbers.

Both need Earth Engine: export `EE_PROJECT` (or pass `project=`) before the first call.

## Fetch a cube

```python
import geopandas as gpd
from loguru import logger

from water_timeseries.utils.earthengine import (
    cached_get_rioxarray_ds_from_lake,
    get_rioxarray_ds_from_lake,
    initialize_earth_engine,
)

initialize_earth_engine()

gdf = gpd.read_parquet("tests/data/lake_polygons.parquet")
lake_id = gdf["id_geohash"].iloc[0]

ds = get_rioxarray_ds_from_lake(
    gdf,
    lake_id,
    "2024-06-15",
    "2024-08-31",
    max_cloud_cover=40,
    buffer=150,          # metres around the polygon, so the shoreline is visible
    bands=("B3", "B4", "B8"),
)
# dims: (time, y, x); B3/B4/B8 plus SCL and MSK_CLDPRB for cloud filtering
```

| argument | meaning |
|---|---|
| `start_date` / `end_date` | inclusive acquisition window (`YYYY-MM-DD`) |
| `max_cloud_cover` | drops acquisitions above this cloud percentage (`"pixel"` mode uses `MSK_CLDPRB`) |
| `buffer` | metres of context around the polygon |
| `bands` | bands to retrieve; `SCL` and `MSK_CLDPRB` are added automatically in `"pixel"` mode |
| `date_windows` | optional `[(start, end), …]` to fetch only imagery near the dates of interest |
| `grid_scale` | output pixel size in metres (10 = native S2; 20–30 for thumbnails) |
| `id_field` | name of the identifier column, default `"id_geohash"` (see [Object Identifiers](id_field.md)) |

`cached_get_rioxarray_ds_from_lake` is the same call behind `st.cache_resource`, keyed by
identifier + query parameters — use it in the dashboard, not in a script.

## Render

```python
from water_timeseries.utils.earthengine import visualize_s2_xee_cube

fig = visualize_s2_xee_cube(ds, ["2024-07-23", "2024-08-23"], style="cir", max_cols=2)
fig.savefig("lake_cir.png", dpi=150, bbox_inches="tight")
```

`visualize_s2_first_and_last(ds, style=...)` is the two-panel shorthand for the first and
last acquisition in the cube.

### Stretches

| `style` | bands | divisor | saturates at |
|---|---|---|---|
| `"rgb"` | B4, B3, B2 | **1000** | 0.10 reflectance |
| anything else (`"cir"`) | B8, B4, B3 | **4000** | 0.40 reflectance |

Values are divided and then clipped to `0–1`. The CIR divisor is larger because near
infrared is far brighter than the visible bands: measured on one summer lake scene,
`/3000` clipped 30–50 % of B8 pixels per date, `/4000` clips 4–20 %.

The divisor is *absolute*, so a pixel at 2000 DN always renders at the same brightness —
that is what makes two dates comparable. A percentile stretch adapts per scene and looks
punchier, but per-image limits make panels incomparable; if you want percentiles, compute
them once per band over the whole cube, not per panel.

### Which acquisition appears in a panel

Each requested date is resolved by `get_nearest_date` (nearest acquisition in the cube),
then `get_better_image` keeps the frame with the fewest missing `B4` pixels. So a panel
labelled *2024-07-23* is that day's least-noisy acquisition, not a composite. Pass
`exact_dates=True` to require the exact date instead of the nearest.

## Notes and limits

- The functions return a `matplotlib` figure; saving is up to the caller.
- Radiometry is left as delivered by Earth Engine: surface reflectance on the 0–10000 DN
  scale, no rescaling before the stretch.
- Snow and cloud are not masked for display; a snowy scene will read bright in CIR. That
  is the point when the question is "was that an ice artefact", and a nuisance otherwise.
