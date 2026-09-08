# Refreshing the dashboard for a new NRT month

When a new month of near-real-time data lands, four separate artifacts have to be
refreshed and the dashboard config has to be re-pointed at two of them. Doing only
some of them fails quietly rather than loudly: the month simply does not appear in
the sidebar, or it appears with the wrong colors, or its lakes render but do not
respond to clicks.

| Artifact | What it drives | Refreshed by |
|---|---|---|
| Dynamic World cube (`.zarr`) | The per-lake timeseries panel | Download from GCS, update `dw_dataset_file` |
| NRT run tiles (`.pmtiles`) | The base lake polygons on the map | Download from GCS, update `pmtiles_file` |
| NRT run geometries (`.parquet`) | Resolving a map click to a lake | Download, `repartition-parquet`, update `vector_file` |
| Breaks table (`nrt_monthly_drain_breaks.parquet`) | Which lakes drained, and their confidence | `merge-nrt-confidence` |
| Month tileset (`nrt_<month>_drainage.pmtiles`) | The Drainage Status overlay | `build-nrt-pmtiles` |

The historical mode is untouched by an NRT month — it reads its own
`precomputed_historical` directory and its own tileset.

## 0. See what upstream actually has

Everything below is gated on what the upstream pipeline has published, so start
here. Both listings need `gcloud auth login` first (interactive, in your own
browser).

```bash
# The imagery-derived cube -- the ceiling on which months can have any NRT data
gcloud storage ls gs://pdg-storage-default/workflows_optimization/lake_change_detection/

# The monthly NRT pipeline runs
gcloud storage ls gs://pdg-storage-default/workflows_optimization/dashboard_nrt/
```

The cube lags real time by roughly a month, and it is the hard ceiling: if the
newest cube stops at 2026-07, there is no 2026-08 data to show no matter what else
exists. A month with a cube but **no** `DW_NRT_<month>_run<date>_allGeoms_v*.parquet`
has no drainage confidence to extract, and the only route to one is the ~30 h/month
`breakpoint-analysis-nrt` batch (see [No full run for the month](#no-full-run-for-the-month)).

Note the exact `run<date>` suffix — it is stamped upstream and is not derivable from
the month. 2026-06's run is `run2025-06-25`; 2026-07's is `run2026-07-31`.

## 1. Dynamic World cube

```bash
gcloud storage cp -r \
    gs://pdg-storage-default/workflows_optimization/lake_change_detection/lakes_dw_V2d_2016-<month>_gapfilled_chunked.zarr \
    data/
```

Then point `dw_dataset_file` at it in `configs/dashboard_panarctic.yaml` (top level —
both modes share the cube).

Prefer the `_gapfilled_chunked` variant. A cube published without that suffix has not
been gap-filled or re-chunked, so the timeseries panel will be slower and may show
holes; it works, but it is not what the config is tuned for.

Leave `dw_start_month` / `dw_end_month` alone. Those are the **seasonal window**
(June–September), not a data cutoff — the dashboard reads them as
`range(dw_start_month, dw_end_month + 1)`. Only `dw_end_year` moves, and only when
the year rolls over.

## 2. NRT run: base tiles and lake geometries

Two files from the same run, both multi-GB (2026-07: 2.7 GB parquet, 4.0 GB tiles):

```bash
mkdir -p data/DW_NRT/DW_NRT_<month>_run<date>
gcloud storage cp \
    "gs://pdg-storage-default/workflows_optimization/dashboard_nrt/DW_NRT_<month>_run<date>_allGeoms_v*.parquet" \
    "gs://pdg-storage-default/workflows_optimization/dashboard_nrt/DW_NRT_<month>_run<date>_allGeoms_v*.pmtiles" \
    data/DW_NRT/DW_NRT_<month>_run<date>/
```

The parquet needs repartitioning before the dashboard can read single lakes out of it
efficiently — sorted by `id_geohash` with small row groups, so a per-lake read touches
one row group instead of scanning ~1M rows:

```bash
uv run water-timeseries repartition-parquet \
    data/DW_NRT/DW_NRT_<month>_run<date>/DW_NRT_<month>_run<date>_allGeoms_v3.parquet \
    data/DW_NRT/DW_NRT_<month>_run<date>/DW_NRT_<month>_run<date>_allGeoms_v3_repartitioned.parquet
```

Then update the `nrt_drainage` mode in `configs/dashboard_panarctic.yaml`:

```yaml
modes:
  nrt_drainage:
    vector_file: data/DW_NRT/DW_NRT_<month>_run<date>/DW_NRT_<month>_run<date>_allGeoms_v3_repartitioned.parquet
    pmtiles_file: data/DW_NRT/DW_NRT_<month>_run<date>/DW_NRT_<month>_run<date>_allGeoms_v3.pmtiles
```

`vector_file` is what turns a map click into a lake id. If it still points at an older
run, lakes that only exist in the new run will paint but not respond to clicks — the
map reads as dead rather than as misconfigured.

This step is also the one you can legitimately skip for a while: the Drainage Status
overlay has its own per-month tileset (step 4) and does not depend on the base run
matching the month. Steps 3 and 4 alone will light up a new month on an older base.

## 3. Merge the month's drainage confidence

```bash
uv run water-timeseries merge-nrt-confidence <month> \
    "DW_NRT_<month>_run<date>_allGeoms_v*.parquet" \
    --breaks-file data/precomputed_nrt/nrt_monthly_drain_breaks.parquet
```

Quote the glob so the shell leaves it alone. `--breaks-file` is required here: unlike
`build-nrt-pmtiles`, this command reads top-level config keys only, so
`--config-file configs/dashboard_panarctic.yaml` would not find `precomputed_nrt_dir`
under `modes.nrt_drainage`.

Takes ~15 s — a column-pruned, filtered read over `gcsfs`, not a download of the
multi-GB run. The previous table is backed up to `.bak`, and the month's existing rows
are dropped before the merge, so re-running is safe.

This is what makes the month appear in the sidebar at all: the month list is built
from `analysis_month` in this table, not from the tilesets on disk.

## 4. Build the month's Drainage Status tileset

```bash
uv run water-timeseries build-nrt-pmtiles \
    --breaks-file data/precomputed_nrt/nrt_monthly_drain_breaks.parquet \
    --geometry-file data/DW_historicalbp_simple_merged_breaks_with_allgeoms_v4.parquet \
    --output-dir data/nrt_tiles \
    --months <month> --poly-max-zoom 12
```

**Always pass `--months`.** Without it every month is rebuilt: ~7 minutes instead of
~40 seconds. Needs `tippecanoe` (`brew install tippecanoe`).

Order matters: build this *after* step 3. A tileset baked from a month with no
confidence bakes that absence into its tile properties, and the overlay falls back to
a flat water-loss gradient. The only fix is rebuilding it.

The zoom where the map switches from centroids to polygons is
`POINT_POLY_SWITCH_ZOOM` in `utils/pmtiles_build.py` (currently 8), and the style
layers derive their `minzoom`/`maxzoom` from it. Both layers are baked
`POINT_POLY_OVERLAP_ZOOMS` levels either side of it, so nudging the switch within
that band is a one-line style change; moving it further needs every archive rebuilt
(~7 minutes for all months) or the map draws nothing at the uncovered zooms.

### Rebuilding a base archive

Both base archives were rebuilt on 2026-08-27 and now bake real centroids below the
switch, with slim centroid properties and a 2 MB per-tile cap (see below). Before
that they did not, and the map went blank when zoomed out: they were
built before `_write_features` stamped per-feature zoom ranges, so every centroid
was baked at every zoom with the tileset's maxzoom of 14, and tippecanoe's default
drop rate of 2.5 per level thinned them by ~2.5^14 — 4,026,306 centroids in
`tilestats` against a `dropped_by_rate` of 4,026,295 at z0, one or two dots per
tile. The previous archives are kept alongside as `*.pre-centroids.pmtiles`.

Nothing needs configuring either way: `archive_bakes_low_zoom_centroids` reads the
answer out of each archive's own tippecanoe metadata, and `map_viewer` /
`pmtiles_viewer` gate the polygons only when it is yes. An archive that says no
gets its polygons drawn at every zoom, as it did before the handoff existed, so an
old archive still renders.

To rebuild one, run the builder that matches its tile properties — mixing them up
silently drops the columns the viz styles on:

```bash
# Historical (drainage_year): date_break, date_break_year, water_change_*, *_median
uv run python -c "
from water_timeseries.utils.pmtiles_build import build_pmtiles_drainage_year
build_pmtiles_drainage_year(
    'data/DW_historicalbp_simple_merged_breaks_with_allgeoms_v4.parquet',
    'data/rebuild.pmtiles')"

# NRT (nrt_drainage): date, drainage_confidence, water_*_absolute
uv run python -c "
from water_timeseries.utils.pmtiles_build import build_pmtiles_nrt_drainage
build_pmtiles_nrt_drainage(
    'data/DW_NRT/.../DW_NRT_2026-06_run2025-06-25_allGeoms_v3_repartitioned.parquet',
    'data/rebuild.pmtiles')"
```

Each takes ~30 minutes on a laptop for 4M lakes (~6 min of that exporting GeoJSONL,
the rest tippecanoe) and needs ~15 GB of scratch space on top of the ~3 GB result.

Two settings decide how much of the map survives being zoomed out, because
`--drop-densest-as-needed` discards lakes until each tile fits the byte cap:
`--maximum-tile-bytes` in `DEFAULT_TIPPECANOE_ARGS` (2 MB, four times the
tippecanoe default) and the per-builder `*_POINT_PROPERTIES`, which keep the
centroids down to the id plus the column the mode colours by. Measured in one z7
region holding 3,052 lakes:

| zoom | 7 props, 500 KB | slim, 500 KB | slim, 2 MB (shipped) |
|-----:|----------------:|-------------:|---------------------:|
| 6    | 32%             | 60%          | 100%                 |
| 5    | 9%              | 26%          | 42%                  |
| 4    | 6%              | 11%          | 19%                  |

The drained lakes escape this entirely. Only ~9,800 lakes in the whole record have
a break date -- 0.2%, a quarter of a single NRT month -- so they get their own
tileset built the way the NRT monthly overlay is, with no size or feature limit and
nothing dropped, drawn over the sampled grey base. That is what keeps them complete
at every zoom (3% -> 100% at z0), which the base archive cannot manage at any cap:

```bash
uv run python -c "
from water_timeseries.utils.pmtiles_build import (
    build_pmtiles_historical_drained, historical_drained_tiles_path)
build_pmtiles_historical_drained(
    'data/DW_historicalbp_simple_merged_breaks_with_allgeoms_v4.parquet',
    historical_drained_tiles_path('data/DW_historicalbp_simple_merged_breaks_with_allgeoms_v4.pmtiles'))"
```

Seconds to build, ~24 MB. **Rebuild it whenever the base archive is rebuilt** -- the
dashboard finds it next to the base archive as `<base>_drained.pmtiles`, so a stale
one is picked up silently and shows the previous run's drained lakes. Delete it (or
point `drained_pmtiles_file:` at nothing) and the drained lakes come out of the base
archive behind a filter instead, as they did before, sampling and all.

Going past 2 MB is not worth it — at 8 MB the tiles stop growing (~2 MB) because a
different limit binds, and coverage barely moves. Below z6 the map is still a
sample: 4M centroids cannot fit in one z4 tile at any cap. The NRT *monthly*
overlay escapes this entirely by being small enough (tens of thousands of
features) to build with `--no-tile-size-limit --no-feature-limit`, which is why
its drained lakes are all present at every zoom.
Build to a new path, check it, then swap it in — the dashboard keeps serving the
old one meanwhile:

```bash
uv run python -c "
from water_timeseries.utils.pmtiles_reader import read_pmtiles_metadata
from water_timeseries.utils.pmtiles_build import archive_bakes_low_zoom_centroids
print(archive_bakes_low_zoom_centroids(read_pmtiles_metadata('<archive>.pmtiles')))"

# and that each zoom draws lakes exactly one way (z0-6 points, z7-9 both, z10+ polys)
tippecanoe-decode <archive>.pmtiles <z> <x> <y> | grep -o '"layer": "[a-z_]*"' | sort | uniq -c
```

## 5. Verify

```bash
uv run python -c "
import pandas as pd
print(pd.read_parquet('data/precomputed_nrt/nrt_monthly_drain_breaks.parquet',
      columns=['analysis_month','drainage_confidence']).groupby('analysis_month').count())"

ls -la data/nrt_tiles/nrt_<month>_drainage.pmtiles
```

Then launch and switch to Near Real-Time:

```bash
uv run water-timeseries dashboard --config-file configs/dashboard_panarctic.yaml
```

The month should appear in the sidebar's "Drainage Status" list with its drained-lake
count, and the log should carry `NRT monthly tiles for <month>: ...`. That log line is
the one to check — without it the overlay silently fell back to the slower runtime
feature-state path, which means step 4's archive was not found.

## 6. Deploying the refresh

In production the config is mounted at `/data/dashboard-config.yaml` and the tiles are
served by the `serve-tiles` sidecar, with `PMTILES_BASE_URL` pointing the browser at
it. Two things follow:

- `serve-tiles` reads only each mode's `pmtiles_file` out of the config — it does
  **not** pick up `nrt_pmtiles_dir`. It serves the parent directory of the archive it
  was given, so per-month archives are reachable by bare filename only if they sit in
  that same directory.
- With `PMTILES_BASE_URL` set, the dashboard asks for `<base_url>/nrt_<month>_drainage.pmtiles`
  — a bare filename, no directory component.

So either copy the month archives next to the base `.pmtiles` on the PVC, or set
`nrt_pmtiles_dir` to an `http(s)://` or `gs://` prefix and let the browser fetch them
straight from there.

## No full run for the month

If GCS has a cube but no `DW_NRT_<month>_run<date>` files, the confidence has to be
computed locally — roughly 30 hours per month of ARIMA fitting:

```bash
uv run water-timeseries breakpoint-analysis-nrt <args>
uv run water-timeseries aggregate-nrt <nrt-dir>
```

`aggregate-nrt` consolidates the per-month `nrt_*_drain_breaks.parquet` files into
`nrt_monthly_drain_breaks.parquet` plus `nrt_monthly_drain_counts.parquet`, which is
the same table step 3 would have written. Steps 4 onward are unchanged.

A month is not always one file in one directory: a tiled run writes each tile's
parquet next to its `.nc`, in that month's own subdirectory. `aggregate-nrt` takes
any number of input directories, so pass them all, or point it at the parent with
`--recursive`:

```bash
# one directory per tile
uv run water-timeseries aggregate-nrt <dir-a> <dir-b> <dir-c> --output-dir <nrt-dir>

# or let it walk the month subdirectories itself
uv run water-timeseries aggregate-nrt <parent-dir> --recursive --output-dir <nrt-dir>
```

Files are deduplicated by resolved path, so overlapping arguments (a parent with
`--recursive` plus one of its own children) cannot double-count a month's rows.
Note that `--output-dir` defaults to the *first* input directory, which is rarely
what you want with several inputs — pass it explicitly.

Always check GCS for a newer full run before committing to this — the run may simply
not have been published yet.
