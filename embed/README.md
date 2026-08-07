# Parent-site embed helper

`water-timeseries-embed.js` is a drop-in snippet for a parent page embedding
the Water Timeseries dashboard with shareable deep links. It forwards `wt_*`
params from the parent URL into the iframe on load, and mirrors the
dashboard's live state back onto the parent URL so links (address bar or the
dashboard's "Copy link" button) restore the exact embedded state.

Full documentation, including a local test setup with a sample parent page:
[docs/embedding.md](../docs/embedding.md).
