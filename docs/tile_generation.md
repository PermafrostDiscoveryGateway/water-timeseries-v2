# Building Map Tilesets

The dashboard's map does not read the lake parquet in the browser. It reads three
PMTiles archives, baked ahead of time by `tippecanoe`, and every lake you see is a
feature in one of them. This page is the reference for how those archives are made:
what goes in, which command bakes which one, and how to check the result before it
goes live. For the monthly routine that calls these commands, see
[Refreshing the dashboard for a new NRT month](nrt_monthly_update.md).

**Prerequisite:** [tippecanoe](https://github.com/felt/tippecanoe) v2.17+ on your
PATH (`brew install tippecanoe`). Every build command fails immediately if it is
missing, before doing any of the slow work.

## The three archives

| Archive | Holds | Built by | Cost | Rebuild when |
|---|---|---|---|---|
| **Shared base** `lakes.pmtiles` | Every lake polygon (~4.03M), flat grey in both modes | `build-pmtiles` | ~35 min, ~2.6 GB | The lake geometry itself changes |
| **Historical overlay** `lakes_drained.pmtiles` | The ~9,800 lakes with a break date, colored by year | `build-drained-pmtiles` | seconds, ~24 MB | The historical breaks change |
| **NRT monthly** `nrt_<month>_drainage.pmtiles` | One archive per month: that month's drained lakes, colored by confidence | `build-nrt-pmtiles` | ~40 s per month | Every new NRT month |

Both view modes render their grey lakes from the *same* base archive, and each
mode draws its own overlay on top of it.

### Why three archives and not one

The base archive holds 4M polygons, which do not fit in a tile at low zoom at any
size cap. `--drop-densest-as-needed` therefore throws lakes away as you zoom out —
and it has no idea which ones matter, so it discards drained lakes along with
stable ones. At z5 roughly a quarter of a dense region survives. That is fine for
the grey backdrop and fatal for the lakes the map exists to show.

The drained lakes are few enough to escape the problem entirely: ~9,800 for the
whole historical record, and tens of thousands for a busy NRT month. Split into
their own archives they build with **no size limit and no feature limit** —
nothing is dropped, at any zoom — and get drawn over the thinned grey base. The
overlays are also orders of magnitude cheaper to rebuild than the base, which is
what makes a monthly refresh a 40-second job instead of a 35-minute one.

## Inputs

Everything starts from the lake tables in `data/`:

| Input | Feeds | Needs |
|---|---|---|
| `DW_historicalbp_simple_merged_breaks_with_allgeoms_v4.parquet` | Base archive, historical overlay, and the geometry source for NRT months | `id_geohash`, `geometry`, the area/change columns, `date_break*` |
| `nrt_monthly_drain_breaks.parquet` | NRT monthly archives | `analysis_month`, `id_geohash`, `drainage_confidence` and the per-month water columns |

The NRT builder takes geometry and drainage signal from two *separate* files: the
breaks table says which lakes drained in a month and carries the values, and the
geometry table supplies the polygons, matched on `id_geohash`. Either mode's lake
table works as that geometry source — both cover the same 4,026,306 lakes with the
same ids and byte-identical geometry — but the historical one is canonical, and is
the only one carrying `date_break_year`, which the historical overlay filters on.

## 1. The shared base archive

```bash
uv run water-timeseries build-pmtiles \
    data/DW_historicalbp_simple_merged_breaks_with_allgeoms_v4.parquet \
    data/lake_geometry/lakes.pmtiles
```

or, reading paths out of a dashboard config:

```bash
uv run water-timeseries build-pmtiles --config-file configs/dashboard_panarctic.yaml
```

The config's `pmtiles_file` key doubles as the output path, so the same file that
serves the dashboard can build its tiles. A multi-mode config builds one mode at a
time; `--mode` picks which, defaulting to `default_mode`.

**Layers:** `lakes` (polygons) and `lakes_points` (centroids).

**Properties:** the id plus the four area/change columns, and `date_break_year`.
Neither mode *styles* from these — the base lakes are flat grey in both — but a
stable lake appears in no overlay, so its base tile is the only thing it can
hover. Stripping the base down to the id alone left millions of lakes with an
empty popup. Centroids carry the id alone: hover is gated off below the switch
zoom, and every property byte spent down there costs another lake to
`--drop-densest-as-needed`.

## 2. The historical drained overlay

```bash
uv run water-timeseries build-drained-pmtiles --config-file configs/dashboard_panarctic.yaml
```

or explicitly:

```bash
uv run water-timeseries build-drained-pmtiles \
    --vector-file data/DW_historicalbp_simple_merged_breaks_with_allgeoms_v4.parquet \
    --pmtiles-file data/lake_geometry/lakes.pmtiles
```

Every lake with a non-null `date_break_year`, filtered out of the same parquet the
base archive is built from. The rows are filtered *before* the WKB geometry is
decoded — parsing 4M geometries to keep 9,800 is most of the runtime of a subset
build, and none of it is needed.

**Output path matters.** The dashboard looks for the overlay at
`<base>_drained.pmtiles`, which is the default when `--pmtiles-file` is given and
`--output-file` is not. The config names it `drained_pmtiles_file`, and a
config-driven build writes exactly there.

!!! warning "The overlay is not optional"
    If it is missing, `drainage_year` mode falls back to filtering the base
    archive on `date_break_year`. The filter matches nothing useful, every lake
    renders as a stable grey dot under a "Drainage Year" legend, and the map looks
    like data rather than a misconfiguration. `app.py` logs a warning; that
    warning is the only signal.

**Layers:** `drained` and `drained_points` — the same names the NRT monthly
archives use, because the style treats the two the same way.

## 3. The NRT monthly archives

```bash
uv run water-timeseries build-nrt-pmtiles \
    --breaks-file data/precomputed_nrt/nrt_monthly_drain_breaks.parquet \
    --geometry-file data/DW_historicalbp_simple_merged_breaks_with_allgeoms_v4.parquet \
    --output-dir data/nrt_tiles \
    --months 2026-08 --poly-max-zoom 12
```

One archive per month, named `nrt_<month>_drainage.pmtiles`. The dashboard layers
the selected month's archive over the base tiles, so switching months is a
source-URL swap: no per-lake data is pushed into the browser on every rerun.

**Always pass `--months`.** Without it every month in the breaks table is rebuilt —
minutes instead of seconds.

**Build it after the month's confidence is merged.** A tileset baked from a month
with no `drainage_confidence` bakes that absence into its tile properties, and the
overlay silently falls back to a flat water-loss gradient. Rebuilding is the only
fix.

**`--poly-max-zoom` is the size lever.** The top zooms dominate an archive: for
2026-07, z13+z14 alone were 43% of the file. Dropping to 12 roughly halves the
polygon layer (96.6 → 48.6 MB) and costs only coordinate quantization above z12 —
MapLibre overzooms the z12 tiles, which already carry full-resolution geometry
(919 vertices in a sample tile at 12, versus 920 at 14).

Months with no tileset fall back to the runtime feature-state path, which is
slower and pushes per-lake values into the page.

### The `scored` layer

!!! warning "The deployed monthly archives carry a layer this branch cannot build"
    The 2026-06 and 2026-07 archives in `data/nrt_tiles/` hold a third layer,
    `scored`, alongside `drained`/`drained_points`. It carries the month's own
    prediction for **every lake the run scored**, not just the ones that drained —
    an observed area, a prediction, an interval and a confidence — under
    `water_*_absolute` column names. It is what lets a *non*-drained lake hover
    month-specific values instead of falling through to the base archive's
    historical area columns.

    That layer is built by `nrt-monthly/12-nrt-scored-layer`, which is **not merged
    into `main`**. It adds `NRT_SCORED_TILE_PROPERTIES`, `nrt_scored_rows`,
    `find_nrt_run_parquets` and a `tile-join` step that merges a scored sidecar
    into the month's archive, driven by a `--nrt-run-dir` flag. Building the layer
    needs a full NRT run to read per-lake predictions from; the breaks table alone
    only knows about drained lakes.

    **Consequence:** rebuilding a month with `build-nrt-pmtiles` as it exists on
    `main` silently drops `scored`. Measured on 2026-07, the archive goes from
    1.34 GB to 60 MB — most of the size *is* that layer — and NRT hover regresses
    for every non-drained lake. Do not overwrite a deployed monthly archive from
    `main`; build from that branch, or build to a new path and compare layers
    before swapping.

A month with no full run to score from keeps the drained layers alone and is
unaffected either way.

## How the layers fit together on the map

Below `POINT_POLY_SWITCH_ZOOM` (currently 8, in `utils/pmtiles_build.py`) a lake is
drawn as a centroid dot; at and above it, as a polygon. Four things have to agree
on that number: the per-feature zoom ranges baked here, the style's circle
`maxzoom`, the style's polygon `minzoom`, and the hover gate — the last three all
in `map_utils.build_pmtiles_map`.

Both layers are baked `POINT_POLY_OVERLAP_ZOOMS` levels either side of the switch,
so retuning it anywhere inside that band is a style change rather than a rebuild of
every archive. Moving it further needs every archive rebuilt, or the map draws
nothing at the uncovered zooms.

The zoom split is enforced by a per-feature `tippecanoe` property
(`{"minzoom": …, "maxzoom": …}`) written onto each GeoJSON feature — **not** by
`minzoom`/`maxzoom` keys in a `-L` layer spec. Those are not part of tippecanoe's
`-L` JSON schema (only `file`, `layer`, `description`, `format` are) and are
silently ignored: with only those set, the `drained` polygon layer showed up at z0
anyway.

## Verifying an archive

Build to a new path, check it, then swap it in — the dashboard keeps serving the
old archive meanwhile.

Read what a finished archive claims about itself:

```python
from water_timeseries.utils.pmtiles_reader import read_pmtiles_header, read_pmtiles_metadata

read_pmtiles_header("data/lake_geometry/lakes.pmtiles")   # zoom range, bounds, tile count
read_pmtiles_metadata("data/lake_geometry/lakes.pmtiles")  # vector_layers, tilestats, strategies
```

Both have `_remote` variants that read an `http(s)://` archive over Range requests,
so a hosted archive can be checked without downloading it.

**Check the centroid handoff survived.** An archive whose centroids were
rate-dropped below the switch zoom cannot back the point-to-polygon handoff — the
map goes blank below the switch. `strategies` in the metadata records that, so the
archive answers for itself:

```python
from water_timeseries.utils.pmtiles_build import archive_bakes_low_zoom_centroids

archive_bakes_low_zoom_centroids(read_pmtiles_metadata(path))  # True if the handoff works
```

Archives built before the per-feature zoom ranges baked every centroid at every
zoom, so tippecanoe's 2.5-per-level drop rate thinned 4M of them down to one or two
dots per tile. Those return `False`, and drawing polygons at every zoom is what
they support.

!!! warning "`vector_layers` misreports per-layer zoom"
    The `minzoom`/`maxzoom` in an archive's `vector_layers` metadata does not
    reflect the per-feature zoom ranges. To confirm which layer is actually
    present at a given zoom, decode a tile:
    `tippecanoe-decode -z<zoom> -x<x> -y<y> archive.pmtiles`.

Keep the intermediate GeoJSONL to inspect what was fed to tippecanoe: every build
command takes `--keep-geojsonl`, which leaves the `.geojsonl` (and
`_points.geojsonl`) files next to the output instead of deleting them.

## Publishing

Upload the finished `.pmtiles` files to object storage (S3, GCS) and point the
dashboard at the public URL with `--pmtiles-url`, or use `--pmtiles-file` for local
development, where the dashboard starts a small HTTP server with Range support.
`nrt_pmtiles_dir` also accepts an `http(s)://` or `gs://` prefix.

For testing MapLibre outside Streamlit, `water-timeseries serve-tiles
data/lake_geometry/lakes.pmtiles --port 8080` serves an archive on its own.
