// vvv THOG
"use strict";

// Tell the server when every weights panel needs only the newest optimizer-step
// snapshot. An explicit retained-step selection takes precedence: request exactly
// that retained step/window and let the browser render every snapshot in it.
(() => {
  const base_fetch_json_current_weights = fetch_json;

  const all_weight_charts_current_only = () => (
    Array.isArray(depth_weight_chart_names)
    && depth_weight_chart_names.length > 0
    && depth_weight_chart_names.every(chart_name => (
      normalize_chart_settings(chart_name)?.current_weights_only === true
    ))
  );

  const selected_step_range = () => {
    const range = window.__instra_weight_step_filter?.request_range?.();
    const minimum = Number(range?.minimum);
    const maximum = Number(range?.maximum);
    if (!Number.isInteger(minimum) || !Number.isInteger(maximum)) return null;
    if (minimum < 0 || maximum < minimum) return null;
    return {minimum, maximum};
  };

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
    ) {
      const step_range = selected_step_range();
      if (step_range) {
        parsed.searchParams.delete("current_only");
        parsed.searchParams.set("step_min", String(step_range.minimum));
        parsed.searchParams.set("step_max", String(step_range.maximum));
        return base_fetch_json_current_weights(
          `${parsed.pathname}?${parsed.searchParams.toString()}`,
          options,
        );
      }
      if (all_weight_charts_current_only()) {
        parsed.searchParams.set("current_only", "1");
        return base_fetch_json_current_weights(
          `${parsed.pathname}?${parsed.searchParams.toString()}`,
          options,
        );
      }
    }
    return base_fetch_json_current_weights(url, options);
  };
})();
// ^^^ THOG
