/**
 * Water Timeseries embed helper for parent sites.
 *
 * Enables shareable deep links when the dashboard is embedded in an iframe:
 *  - On load, forwards wt_-prefixed state params from the parent page URL into
 *    the iframe src (so a pasted link restores the embedded dashboard state).
 *  - Listens for state updates from the dashboard and mirrors them onto the
 *    parent page URL via history.replaceState (so the address bar and the
 *    dashboard's "Copy link" button always reflect the current state).
 *
 * Usage:
 *   <iframe id="wt-frame" allow="clipboard-write"></iframe>
 *   <script src="water-timeseries-embed.js"></script>
 *   <script>
 *     WaterTimeseriesEmbed.init({
 *       iframe: "#wt-frame",
 *       appUrl: "https://your-dashboard.example.org",
 *     });
 *   </script>
 *
 * The iframe MUST carry allow="clipboard-write" for the dashboard's copy
 * button to reach the clipboard directly (it falls back to a selectable text
 * field otherwise).
 */
(function (global) {
  "use strict";

  var DEFAULT_PREFIX = "wt_";

  function init(options) {
    if (!options || !options.appUrl) {
      throw new Error("WaterTimeseriesEmbed.init: appUrl is required");
    }
    var iframe =
      typeof options.iframe === "string"
        ? document.querySelector(options.iframe)
        : options.iframe;
    if (!iframe) {
      throw new Error("WaterTimeseriesEmbed.init: iframe not found");
    }
    var prefix = options.prefix || DEFAULT_PREFIX;
    var appOrigin = new URL(options.appUrl, global.location.href).origin;

    // 1) Forward wt_* params from the parent URL into the iframe src.
    var appUrl = new URL(options.appUrl, global.location.href);
    var pageParams = new URLSearchParams(global.location.search);
    pageParams.forEach(function (value, key) {
      if (key.indexOf(prefix) === 0) {
        appUrl.searchParams.set(key.slice(prefix.length), value);
      }
    });
    if (!appUrl.searchParams.has("embed")) {
      appUrl.searchParams.set("embed", "true");
    }
    iframe.src = appUrl.toString();

    // 2) Answer the dashboard's handshake and mirror its state onto our URL.
    global.addEventListener("message", function (event) {
      if (event.origin !== appOrigin) return;
      var data = event.data;
      if (!data || typeof data !== "object") return;

      if (data.type === "wt:hello") {
        event.source.postMessage(
          { type: "wt:hello-ack", version: 1, href: global.location.href, prefix: prefix },
          event.origin
        );
      } else if (data.type === "wt:state") {
        var url = new URL(global.location.href);
        var stale = [];
        url.searchParams.forEach(function (value, key) {
          if (key.indexOf(prefix) === 0) stale.push(key);
        });
        stale.forEach(function (key) {
          url.searchParams.delete(key);
        });
        var params = data.params || {};
        Object.keys(params).forEach(function (key) {
          url.searchParams.set(prefix + key, params[key]);
        });
        global.history.replaceState(null, "", url.toString());
      }
    });
  }

  global.WaterTimeseriesEmbed = { init: init };
})(window);
