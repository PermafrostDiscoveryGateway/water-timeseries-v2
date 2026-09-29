# Refreshing the dashboard for a new NRT month

When a new month of near-real-time data lands, a handful of artifacts have to be
refreshed. Doing only some of them fails quietly rather than loudly: the month simply
does not appear in the sidebar, or it appears with the wrong colors, or its lakes
render but do not respond to clicks.

**Every month:**

| Artifact | What it drives | Refreshed by | Cost |
|---|---|---|---|
| Dynamic World cube (`.zarr`) | The per-lake timeseries panel | Download from GCS, update `dw_dataset_file` | multi-GB download |
| Breaks table (`nrt_monthly_drain_breaks.parquet`) | Which lakes drained, and their confidence | `merge-nrt-confidence` | ~15 s |
| Month tileset (`nrt_<month>_drainage.pmtiles`) | The Drainage Status overlay, and what every lake hovers | `build-nrt-pmtiles --months` | ~40 s, or 8-12 min with a full run to score from |

**Only when it has fallen behind** (deferrable — see step 2):

| Artifact | What it drives | Refreshed by | Cost |
|---|---|---|---|
| NRT run geometries (`.parquet`) | Resolving a map click to a lake | Download, `repartition-parquet`, update `vector_file` | multi-GB download |

**Rarely — only when the lake geometry itself changes, not per month:**

| Artifact | What it drives | Refreshed by | Cost |
|---|---|---|---|
| Shared base archive (`data/lake_geometry/lakes.pmtiles`) | The grey lake polygons, in *both* modes | `build-pmtiles` | ~35 min |
| Historical drained overlay (`lakes_drained.pmtiles`) | The colored drained lakes in historical mode | `build-drained-pmtiles` | seconds |

Both modes render their lakes from the *one* shared base archive, so an NRT month no
longer needs its multi-GB `.pmtiles` downloaded or `pmtiles_file` re-pointed — the
modes cover the same lakes with the same geometry (verified byte-identical), and
neither styles the base layer from a baked property.

The historical mode is otherwise untouched by an NRT month — it reads its own
`precomputed_historical` directory and its own drained overlay.

How each of those tilesets is actually built — inputs, layers, tippecanoe profiles, and
why there are three archives rather than one — is in
[Building Map Tilesets](tile_generation.md). This page is the monthly routine; that one
is the reference behind it.

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

The `allGeoms_v<n>` version is stamped upstream too, and it does **not** track the
month: 2026-07 published `allGeoms_v2` while the older 2026-06 run is `allGeoms_v3`.
Read it off the listing rather than assuming; the commands below use a `v*` glob, and
so does `find_nrt_run_parquets`.

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

## 2. NRT run: lake geometries

Only the parquet is needed. The run also publishes a multi-GB `.pmtiles`, but the
dashboard no longer uses it: both modes render from the shared base archive (see
[Rebuilding the base archive and the historical overlay](#rebuilding-the-base-archive-and-the-historical-overlay)), so
downloading the run's tiles and re-pointing `pmtiles_file` is no longer part of a
monthly refresh.

```bash
mkdir -p data/DW_NRT/DW_NRT_<month>_run<date>
gcloud storage cp \
    "gs://pdg-storage-default/workflows_optimization/dashboard_nrt/DW_NRT_<month>_run<date>_allGeoms_v*.parquet" \
    data/DW_NRT/DW_NRT_<month>_run<date>/
```

The parquet needs repartitioning before the dashboard can read single lakes out of it
efficiently — sorted by `id_geohash` with small row groups, so a per-lake read touches
one row group instead of scanning ~1M rows:

```bash
uv run water-timeseries repartition-parquet \
    data/DW_NRT/DW_NRT_<month>_run<date>/DW_NRT_<month>_run<date>_allGeoms_v<n>.parquet \
    data/DW_NRT/DW_NRT_<month>_run<date>/DW_NRT_<month>_run<date>_allGeoms_v<n>_repartitioned.parquet
```

Substitute the `v<n>` the run actually published (see above). Keep the
`_repartitioned` suffix: `find_nrt_run_parquets` prefers that copy when it picks the
month's run for the `scored` layer, and the dashboard reads single lakes out of it.
For 2026-07 this was 2.92 GB downloaded at ~30 MB/s and a 13 s repartition into 2014
row groups.

Then update the `nrt_drainage` mode in `configs/dashboard_panarctic.yaml`:

```yaml
modes:
  nrt_drainage:
    vector_file: data/DW_NRT/DW_NRT_<month>_run<date>/DW_NRT_<month>_run<date>_allGeoms_v3_repartitioned.parquet
    # pmtiles_file is the shared base archive and does not change per month.
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

## 4. Build the month's tileset

```bash
uv run water-timeseries build-nrt-pmtiles \
    --config-file configs/dashboard_panarctic.yaml \
    --months <month> --poly-max-zoom 12
```

The config supplies `breaks_file`, `geometry_file`, `output_dir` and `nrt_run_dir`
(the last one is what gets the `scored` layer built — see below); pass them as flags
instead if you are working outside a config:

```bash
uv run water-timeseries build-nrt-pmtiles \
    --breaks-file data/precomputed_nrt/nrt_monthly_drain_breaks.parquet \
    --geometry-file data/DW_historicalbp_simple_merged_breaks_with_allgeoms_v4.parquet \
    --output-dir data/nrt_tiles --nrt-run-dir data/DW_NRT \
    --months <month> --poly-max-zoom 12
```

**Always pass `--months`.** Without it every month in the breaks table is rebuilt —
drained layers are ~10 s each (~7 minutes across a 41-month table), but every month
that also has a full run under `nrt_run_dir` adds 8-12 minutes of its own. Needs
`tippecanoe` (`brew install tippecanoe`).

### What the archive holds, and why `--nrt-run-dir`

Two layers, because two sets of lakes have two different kinds of value:

| Layer | Lakes | Values | Budget |
|---|---|---|---|
| `drained` (+ `drained_points`) | drained that month (~36k-61k) | the drainage signal | every feature kept, at every zoom |
| `scored` | every lake the month's full run predicted for | observed vs predicted area, CI, residual, confidence | `--drop-densest-as-needed`, 2 MB tiles |

The `scored` layer is what a **non-drained** lake hovers. Without it the month's
tileset spoke only for its drained lakes, and every other lake fell through to
`pmtiles_file` — the shared base archive, which is mode-agnostic and has nothing
month-specific in it, so a stable lake's popup showed the lake id and the historical
area columns and no NRT values at all.

It is built only for the months `--nrt-run-dir` finds a run parquet for
(`DW_NRT_<month>_run<date>/..._allGeoms_v*_repartitioned.parquet`, `_subset` files
ignored), because only a full run predicts per lake: the breaks table has the drained
lakes and nothing else. A month with no run there gets the `drained` layers alone and
works exactly as before, and the dashboard reads which layers a month actually has off
the archive (`pmtiles_has_layer`), so nothing has to be configured per month.

The two layers cannot come out of one tippecanoe run: its density and tile-size flags
are per invocation, not per `-L` layer, and these two need opposite ones (a dropped
drained lake is a lake missing from the map; 1.8M scored lakes have to be dropped to
fit). So they are tiled separately and merged with `tile-join -pk` — `-pk` matters,
since tile-join otherwise re-applies its own 500 KB cap and would drop from the
drained layer that the split build exists to protect.

Measured at `--poly-max-zoom 12`, both months that have a full run:

| Month | Run | Scored lakes | Archive | Build |
|---|---|---|---|---|
| 2026-06 | `run2025-06-25`, `allGeoms_v3` | 1,842,467 (45.8%) | 820.6 MB | ~8 min |
| 2026-07 | `run2026-07-31`, `allGeoms_v2` | 3,027,972 (75.2%) | 1339.7 MB | ~12 min |

Most of that is the scored layer: it streams a ~3 GB intermediate GeoJSONL through
tippecanoe. The drained-only path is still ~40 s, and the drained layer itself is
33.9 MB at `--poly-max-zoom 12` against 57.3 MB at 14.

**Scored coverage is a property of the run, not of this build** — 45.8% of lakes for
2026-06 against 75.2% for 2026-07. So the same lake can hover full NRT values on one
month and only the base archive's historical area columns on another. An unscored lake
has null values and `"nan : nan"` in its CI column, which the popup drops rather than
showing, so this reads as a shorter popup and not as an error.

Order matters: build this *after* step 3. A tileset baked from a month with no
confidence bakes that absence into its tile properties, and the overlay falls back to
a flat water-loss gradient. The only fix is rebuilding it.

The zoom where the map switches from centroids to polygons is
`POINT_POLY_SWITCH_ZOOM` in `utils/pmtiles_build.py` (currently 8), and the style
layers derive their `minzoom`/`maxzoom` from it. Both layers are baked
`POINT_POLY_OVERLAP_ZOOMS` levels either side of it, so nudging the switch within
that band is a one-line style change; moving it further needs every archive rebuilt
(~7 minutes for all months) or the map draws nothing at the uncovered zooms.

### Rebuilding the base archive and the historical overlay

Neither is part of a monthly refresh — rebuild them only when the lake geometry or the
historical breaks change. [Building Map Tilesets](tile_generation.md) covers what each
carries and why; the commands are:

```bash
# The shared base archive both modes render their grey lakes from (~35 min, ~2.6 GB)
uv run water-timeseries build-pmtiles \
    data/DW_historicalbp_simple_merged_breaks_with_allgeoms_v4.parquet \
    data/lake_geometry/lakes.pmtiles

# The historical drained-lakes overlay, which must land beside it (seconds, ~24 MB)
uv run water-timeseries build-drained-pmtiles --config-file configs/dashboard_panarctic.yaml
```

Either mode's table works as the geometry source for the base archive: both cover the
same 4,026,306 lakes with the same `id_geohash` and byte-identical geometry. The
historical one is the canonical choice, and the only one carrying `date_break_year`,
which the drained overlay filters on.

> **The drained overlay is not optional.** Deleting it (or pointing
> `drained_pmtiles_file:` at nothing) falls back to filtering the base archive on
> `date_break_year`, which the shared base archive does not carry — so the filter
> matches nothing, every lake renders as a stable grey dot under a "Drainage Year"
> legend, and the map looks like data rather than like a misconfiguration. `app.py`
> logs a warning when it is missing in `drainage_year` mode; that warning is the only
> signal.

Build to a new path, check it, then swap it in — the dashboard keeps serving the old
one meanwhile. See
[Verifying an archive](tile_generation.md#verifying-an-archive).

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
count, and the log should carry
`NRT monthly tiles for <month>: ... (scored layer: True)`. That log line is the one to
check, in two parts: no line at all means the overlay silently fell back to the slower
runtime feature-state path (step 4's archive was not found), and `scored layer: False`
means the archive is there but was built without a full run for the month, so
non-drained lakes will hover the base archive's historical columns instead of the
month's prediction.

Confirm what actually got baked by reading the archive rather than its file size:

```bash
uv run python -c "
from water_timeseries.utils.pmtiles_reader import read_pmtiles_metadata
md = read_pmtiles_metadata('data/nrt_tiles/nrt_<month>_drainage.pmtiles')
for layer in md['vector_layers']:
    print(layer['id'], sorted(layer['fields']))"
```

Expect `drained`, `drained_points` and — for a month with a full run — `scored`. Tile
metadata is not proof that a layer's features survived at a given zoom, though; for
that, decode tiles (see [Background: the centroid handoff](#background-the-centroid-handoff)).

## 6. Deploying the refresh

In production the config is mounted at `/data/dashboard-config.yaml` and the tiles are
served by the `serve-tiles` sidecar, with `PMTILES_BASE_URL` pointing the browser at
it. Two things follow:

- `serve-tiles` reads only each mode's `pmtiles_file` out of the config — it does
  **not** pick up `nrt_pmtiles_dir`. It serves the parent directory of the archive it
  was given, so per-month archives are reachable by bare filename only if they sit in
  that same directory.
- With `PMTILES_BASE_URL` set, the dashboard asks for `<base_url>/nrt_<month>_drainage.pmtiles`
  — a bare filename, no directory component. The same applies to the historical
  drained overlay: `<base_url>/lakes_drained.pmtiles`. Keeping it beside the shared
  base archive means the directory `serve-tiles` was pointed at already contains it,
  so it resolves with no extra deployment step.

So either copy the month archives next to the base `.pmtiles` on the PVC, or set
`nrt_pmtiles_dir` to an `http(s)://` or `gs://` prefix and let the browser fetch them
straight from there.

**Budget for the size.** A month with a `scored` layer is ~0.8-1.3 GB, not the ~35-60 MB
a drained-only month costs — 820.6 MB for 2026-06 and 1339.7 MB for 2026-07. Two such
months is more than the shared base archive itself (4.1 GB) is worth thinking about
next to. The browser only range-requests the tiles it needs, so this is a storage and
transfer question, not a client-performance one, but a PVC sized for the old archives
will not take many of these. If it matters, `--poly-max-zoom 12` is already applied
above; the next lever is not building `scored` for older months you do not need it on
(just leave their run out of `nrt_run_dir`).

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
