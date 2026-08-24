// vvv THOG
"use strict";

// Route Weights family requests according to the effective per-chart display state.
// This installs before the dashboard performance layer captures fetch_json, so later
// refreshes retain efficient latest-only/range-only server paths without global flags.
window.addEventListener("load", () => {
  const base_fetch_json = fetch_json;

  const selected_range = () => {
    const range = window.__instra_weight_step_filter?.request_range?.();
    const minimum = Number(range?.minimum);
    const maximum = Number(range?.maximum);
    if (!Number.isInteger(minimum) || !Number.isInteger(maximum)) return null;
    if (minimum < 0 || maximum < minimum) return null;
    return {minimum, maximum};
  };

  const current_only_flags = () => (
    Array.isArray(depth_weight_chart_names)
      ? depth_weight_chart_names.map(chart_name => (
          normalize_chart_settings(chart_name)?.current_weights_only === true
        ))
      : []
  );

  fetch_json = async function(url, options = {}) {
    let parsed;
    try {
      parsed = new URL(url, window.location.origin);
    } catch (_error) {
      return base_fetch_json(url, options);
    }
    if (
      options?.method
      || parsed.pathname !== "/api/figure-family"
      || parsed.searchParams.get("family") !== "depth"
    ) {
      return base_fetch_json(url, options);
    }

    const flags = current_only_flags();
    const any_current_only = flags.some(Boolean);
    const all_current_only = flags.length > 0 && flags.every(Boolean);
    const range = selected_range();

    parsed.searchParams.delete("current_only");
    parsed.searchParams.delete("step_min");
    parsed.searchParams.delete("step_max");

    if (all_current_only) {
      parsed.searchParams.set("current_only", "1");
    } else if (range && !any_current_only) {
      parsed.searchParams.set("step_min", String(range.minimum));
      parsed.searchParams.set("step_max", String(range.maximum));
    }
    // Mixed current/history mode intentionally fetches retained history once. The
    // final Weights owner reduces current-only charts to latest and historical charts
    // to the selected range client-side.

    return base_fetch_json(`${parsed.pathname}?${parsed.searchParams.toString()}`, options);
  };

  window.__instra_weight_request_router = Object.freeze({
    current_only_flags,
    selected_range,
  });
});
// ^^^ THOG
