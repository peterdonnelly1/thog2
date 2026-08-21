// vvv THOG
"use strict";

// Tell the server when every weights panel needs only the newest optimizer-step
// snapshot.  The server can then avoid reading, decompressing, building and
// serialising the retained history that the browser would immediately discard.
(() => {
  const base_fetch_json_current_weights = fetch_json;

  const all_weight_charts_current_only = () => (
    Array.isArray(depth_weight_chart_names)
    && depth_weight_chart_names.length > 0
    && depth_weight_chart_names.every(chart_name => (
      normalize_chart_settings(chart_name)?.current_weights_only === true
    ))
  );

  fetch_json = async function(url, options = {}) {
    let parsed = null;
    try {
      parsed = new URL(url, window.location.origin);
    } catch (_error) {
      return base_fetch_json_current_weights(url, options);
    }
    if (
      !options?.method
      && parsed.pathname === "/api/figure-family"
      && parsed.searchParams.get("family") === "depth"
      && all_weight_charts_current_only()
    ) {
      parsed.searchParams.set("current_only", "1");
      return base_fetch_json_current_weights(
        `${parsed.pathname}?${parsed.searchParams.toString()}`,
        options,
      );
    }
    return base_fetch_json_current_weights(url, options);
  };
})();
// ^^^ THOG
