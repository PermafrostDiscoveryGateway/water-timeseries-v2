# Embedding & shareable links

The dashboard keeps its window state in readable URL query params, so any URL
copied from the address bar (or via the sidebar **🔗 Copy link to this view**
button) restores the exact same view when pasted.

## State params

| Param | Format | Meaning |
|---|---|---|
| `selected_lake` | 12-char geohash | Currently selected lake |
| `lat`, `lon` | float (5 decimals) | Map center |
| `zoom` | float (2 decimals) | Map zoom level |
| `drained` | `1` | "Show temporal drainage statistics" toggle is on |
| `month` | `YYYY-MM` | Selected historical-drainage analysis month |
| `hide_stable` | `1` | "Hide stable lakes" toggle is on |

Params at their default value are omitted. Example:

```
https://dashboard.example.org/?selected_lake=b7zpm2xq4k9d&lat=66.512&lon=-164.087&zoom=12&drained=1&month=2024-06
```

The visualization preset (`--viz-configuration`) is fixed per deployment and is
not part of the URL.

## Embed config params

These configure how the app behaves when embedded; they're read once on load
and are never part of the shareable state above (an embedding parent sets
them, they're not echoed back via postMessage or included in copied links).

| Param | Format | Meaning |
|---|---|---|
| `theme` | `light` \| `dark` | Forces the color scheme, overriding the browser's `prefers-color-scheme` |
| `show_share` | `false` | Hides the sidebar **🔗 Copy link to this view** button (e.g. when the embedding parent offers its own shareable link) |

## Embedding in a parent site

When the dashboard is embedded in an iframe, links must point at the *parent*
page, not the framed app. The parent page cooperates via
[`embed/water-timeseries-embed.js`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2/tree/main/embed):

```html
<iframe id="wt-frame" allow="clipboard-write" title="Water Timeseries dashboard"></iframe>

<script src="water-timeseries-embed.js"></script>
<script>
  WaterTimeseriesEmbed.init({
    iframe: "#wt-frame",
    appUrl: "https://your-dashboard.example.org",
  });
</script>
```

The snippet:

1. **On load**, copies `wt_`-prefixed params from the parent URL into the
   iframe `src` (unprefixed), plus `embed=true` — so a pasted parent link
   restores the embedded dashboard state.
2. **Mirrors live state** (`mcui:state` messages) onto the parent URL via
   `history.replaceState` (state params carried with the `wt_` prefix to avoid
   collisions with the parent page's own params), so the parent address bar
   stays shareable as the user pans, zooms, selects lakes, and flips toggles.

Requirements:

- `allow="clipboard-write"` on the iframe tag — without it, the copy button
  cannot write to the clipboard directly and falls back to a selectable text
  field.
- The snippet only accepts messages whose origin matches `appUrl`.

### Locking down the message target

By default the dashboard posts state messages with target origin `*` (they
contain only the state params above — nothing sensitive). To restrict them to
a single parent origin, set an environment variable on the dashboard host:

```bash
WT_PARENT_ORIGIN=https://parent-site.example.org
```

With this set, messages are only delivered to that origin; framed by anyone
else, the message is silently dropped by the browser instead of leaking state.

## postMessage protocol reference

All messages are objects with `type` and `version: 1`.

| Type | Direction | Payload |
|---|---|---|
| `mcui:state` | dashboard → parent (`window.top`) | `{ params }` — current state params; absent keys mean "remove" |

## Local end-to-end test

1. Start the dashboard as usual (it listens on `http://localhost:8501`):

   ```bash
   water-timeseries dashboard --pmtiles-file <tiles.pmtiles> \
       --viz-configuration nrt_drainage --precomputed-nrt-dir <dir>
   ```

2. Create a minimal parent page next to `water-timeseries-embed.js`
   (e.g. `embed/parent.html` — kept out of version control):

   ```html
   <!DOCTYPE html>
   <html>
   <body>
     <iframe id="wt-frame" allow="clipboard-write"
             style="width: 100%; height: 90vh; border: 1px solid #ccc;"></iframe>
     <script src="water-timeseries-embed.js"></script>
     <script>
       WaterTimeseriesEmbed.init({
         iframe: "#wt-frame",
         appUrl: "http://localhost:8501",
       });
     </script>
   </body>
   </html>
   ```

3. Serve the repo over a *different* origin (realistic cross-origin setup)
   and open <http://localhost:8000/embed/parent.html>:

   ```bash
   python -m http.server 8000
   ```

4. Pan, zoom, select a lake, flip toggles — the parent address bar gains
   `wt_*` params live. Click **🔗 Copy link to this view** and paste into a
   fresh tab: the parent page loads with the iframe restored to the same state.
