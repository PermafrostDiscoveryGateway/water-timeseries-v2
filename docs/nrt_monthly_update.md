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
| Shared base archive (`data/lake_geometry/lakes.pmtiles`) | The grey lake polygons, in *both* modes | `build-pmtiles --shared-geometry` | ~35 min |
| Historical drained overlay (`lakes_drained.pmtiles`) | The colored drained lakes in historical mode | `build_pmtiles_historical_drained` | seconds |

Both modes render their lakes from the *one* shared base archive, so an NRT month no
longer needs its multi-GB `.pmtiles` downloaded or `pmtiles_file` re-pointed — the
modes cover the same lakes with the same geometry (verified byte-identical), and
neither styles the base layer from a baked property.

The historical mode is otherwise untouched by an NRT month — it reads its own
`precomputed_historical` directory and its own drained overlay.

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

## 2. NRT run: lake geometries

Only the parquet is needed. The run also publishes a multi-GB `.pmtiles`, but the
dashboard no longer uses it: both modes render from the shared base archive (see
[Rebuilding the shared base archive](#rebuilding-the-shared-base-archive)), so
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
    data/DW_NRT/DW_NRT_<month>_run<date>/DW_NRT_<month>_run<date>_allGeoms_v3.parquet \
    data/DW_NRT/DW_NRT_<month>_run<date>/DW_NRT_<month>_run<date>_allGeoms_v3_repartitioned.parquet
```

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

### Rebuilding the shared base archive

There is one base archive for both modes. Its polygons bake `id_geohash` plus the
four mode-agnostic area/change columns (`Area_start_ha`, `Area_end_ha`,
`NetChange_ha`, `NetChange_perc`) and `date_break_year`; its centroids bake the id
alone. Rebuild it only when the lake geometry changes — an NRT month does not need it:

```bash
uv run water-timeseries build-pmtiles \
    data/DW_historicalbp_simple_merged_breaks_with_allgeoms_v4.parquet \
    data/lake_geometry/lakes.pmtiles --shared-geometry
```

Either mode's table works as the geometry source: both cover the same 4,026,306 lakes
with the same `id_geohash` and byte-identical geometry. The historical one is the
canonical choice.

Takes ~35 minutes on a laptop for 4M lakes (~6 min of that exporting GeoJSONL, the
rest tippecanoe), needs ~10 GB of scratch space, and lands at ~2.6 GB.

Do **not** use the older per-mode builders (`build_pmtiles_drainage_year`,
`build_pmtiles_nrt_drainage`) for the base archive. They bake mode- and month-specific
properties that nothing can read correctly from a shared archive, and on 4M features
that weight costs lakes at low zoom, because `--drop-densest-as-needed` discards
features to fit the byte cap.

#### What the base archive has to carry, and why

Neither mode *styles* from a base property — the base lakes are flat grey in both
(`get_style_pmtiles_stable_lakes`). But the base tile is the last thing a **stable**
lake can hover, and for many of them the only thing. In historical mode the drained
overlay holds drained lakes only, so a stable lake is in no overlay at all. In NRT
mode the month's archive also carries every lake its run scored, but a run scores only
part of the table (45.8% for 2026-06, 75.2% for 2026-07), so the rest still fall
through to here. Baking the id alone therefore left millions of lakes with an empty
popup (the tooltip drops null and the `"NaT"`/`"nan"` placeholders, so a tile with
nothing else on it renders no rows at all).

The four area/change columns are the right thing to share because they describe the
lake rather than a month, and are populated for 100% of stable lakes. Per-month NRT
values are the opposite and must stay in the monthly archives — a single run's
`water_observed_absolute` would be the wrong month's number for every other month,
which is why `build_pmtiles_map` suppresses `date`/`drainage_confidence` from the base
layer. Those values are emphatically *not* empty for non-drained lakes — a stable lake
the run scored has an observed area, a prediction, an interval and a confidence of 0,
which is exactly what the `scored` layer exists to serve. They just have to come from
the month's own archive rather than from a base archive shared by 41 months and two
modes.

`date_break_year` rides along for two reasons: `STABLE_LAKE_FILTER` tests it to tell
the stable and drained layers apart, and it is null for all 4,016,467 stable lakes, so
tippecanoe drops it and it costs bytes only on the ~9,800 drained ones. `date_break`
is deliberately excluded — being a datetime it stringifies to `"NaT"` instead of
dropping, putting a placeholder on 4M features.

The geometry source is itself a local derivative, not a GCS artifact:
`DW_historicalbp_simple_merged_breaks_with_allgeoms_v4.parquet` is
`breakpoint-analysis` output (`output_geometry_all=True`, so all 4,026,306 lakes get
a row), built over the `Nitze_etal_Lakes_filtered_full_set_V2d.parquet` geometry on
the AWI share named in `configs/config.yaml`. **v4 was never uploaded to GCS** —
`gs://pdg-storage-default/workflows_optimization/lake_drainage/DW_historical_drainage`
has v3 only. Don't go looking for v4 in the bucket; the routes are someone's local
copy or a multi-hour regeneration.

#### Background: the centroid handoff

Every current archive bakes real centroids below the switch zoom. Archives built
before `_write_features` stamped per-feature zoom ranges did not — every centroid was
baked at every zoom with the tileset's maxzoom of 14, and tippecanoe's default drop
rate of 2.5 per level thinned them by ~2.5^14 (4,026,306 centroids in `tilestats`
against a `dropped_by_rate` of 4,026,295 at z0, one or two dots per tile), so the map
went blank when zoomed out. The pre-2026-08-27 per-mode archives are kept alongside as
`*.pre-centroids.pmtiles`; they are unreferenced by any config.

Nothing needs configuring either way: `archive_bakes_low_zoom_centroids` reads the
answer out of each archive's own tippecanoe metadata, and `map_viewer` /
`pmtiles_viewer` gate the polygons only when it is yes. An archive that says no
gets its polygons drawn at every zoom, as it did before the handoff existed, so an
old archive still renders.

Two settings decide how much of the map survives being zoomed out, because
`--drop-densest-as-needed` discards lakes until each tile fits the byte cap:
`--maximum-tile-bytes` in `DEFAULT_TIPPECANOE_ARGS` (2 MB, four times the
tippecanoe default) and how few properties get baked — `SHARED_GEOMETRY_TILE_PROPERTIES`
for the shared base (the id alone), or the per-builder `*_POINT_PROPERTIES` for the
overlays (the id plus the column the mode colours by). Measured in one z7
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
    historical_drained_tiles_path('data/lake_geometry/lakes.pmtiles'))"
```

Seconds to build, ~24 MB. It must land **beside the shared base archive** — the
dashboard finds it as `<base>_drained.pmtiles`, i.e.
`data/lake_geometry/lakes_drained.pmtiles`. Rebuild it whenever the historical breaks
change; a stale one is picked up silently and shows the previous run's drained lakes.

> **This overlay is not optional.** Deleting it (or pointing `drained_pmtiles_file:`
> at nothing) used to fall back to filtering the base archive on `date_break_year`.
> The shared base archive does not carry that property, so the filter matches nothing:
> every lake renders as a stable grey dot under a "Drainage Year" legend, and the map
> looks like data rather than like a misconfiguration. `app.py` logs a warning when
> the overlay is missing in `drainage_year` mode — that warning is the only signal.

One trap when relocating it: `drained_pmtiles_file:` in the YAML only reaches a fresh
page load because `cli.py dashboard` forwards it as `--drained-pmtiles-file`. Any new
mode setting needs all three hops — config, `cli.py` `script_args`, and `app.py`
argparse plus the `main()` call — or it silently applies only after a mode switch,
which re-reads the YAML directly. Keeping the overlay at the conventional path means
both routes agree and either one alone is sufficient.

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
  — a bare filename, no directory component. The same applies to the historical
  drained overlay: `<base_url>/lakes_drained.pmtiles`. Keeping it beside the shared
  base archive means the directory `serve-tiles` was pointed at already contains it,
  so it resolves with no extra deployment step.

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
