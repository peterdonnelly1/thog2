// vvv THOG
"use strict";

// Legacy heatmaps without centre-loss metadata cannot support percentage Δloss.
// Preserve their retained raw Δloss in absolute mode without owning any Weights state.
window.addEventListener("load", () => {
  const finite_number = value => {
    if (value === null || value === undefined || value === "") return null;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
  };

  const clamp_01 = value => Math.max(0, Math.min(1, Number(value)));
  const viewer_limit = (name, fallback) => {
    const value = finite_number(heatmap_settings_for_current_run()?.[name]);
    return value !== null && value > 0 ? value : fallback;
  };
  const manual_band_limits = () => {
    const base_limit = heatmap_abs_limit(0.05);
    return {
      green: Math.min(0.1, viewer_limit("negative_abs_limit", Math.min(0.1, base_limit))),
      blue: Math.min(1.0, Math.max(0.100000001, viewer_limit("blue_abs_limit", 1.0))),
      yellow: Math.max(1.000000001, viewer_limit("yellow_abs_limit", 2.0)),
      red: viewer_limit("positive_abs_limit", base_limit),
    };
  };
  const limits_from_values = values => {
    const limits = {green: 0, blue: 0, yellow: 0, red: 0};
    for (const value of values) {
      if (!Number.isFinite(value)) continue;
      if (value <= -1.0) limits.yellow = Math.max(limits.yellow, Math.abs(value));
      else if (value <= -0.1) limits.blue = Math.max(limits.blue, Math.abs(value));
      else if (value < 0) limits.green = Math.max(limits.green, Math.abs(value));
      else if (value > 0) limits.red = Math.max(limits.red, value);
    }
    return limits;
  };
  const band_value = (value, limits) => {
    if (!Number.isFinite(value)) return null;
    if (value <= -1.0) {
      const denominator = Math.max(1e-12, limits.yellow - 1.0);
      return -0.76 - 0.24 * clamp_01((Math.abs(value) - 1.0) / denominator);
    }
    if (value <= -0.1) {
      const denominator = Math.max(1e-12, limits.blue - 0.1);
      return -0.51 - 0.23 * clamp_01((Math.abs(value) - 0.1) / denominator);
    }
    if (value < 0) {
      const intensity = limits.green > 0 ? clamp_01(Math.abs(value) / limits.green) : 1;
      return -0.01 - 0.48 * intensity;
    }
    if (value > 0) {
      const intensity = limits.red > 0 ? clamp_01(value / limits.red) : 1;
      return 0.01 + 0.99 * intensity;
    }
    return 0;
  };
  const format_limit = (value, sign) => (
    Number(value) > 0 ? `${sign}${Number(value).toPrecision(3)}` : "—"
  );

  const percent_requested = () => (
    heatmap_settings_for_current_run()?.delta_loss_display_mode !== "absolute"
  );

  const heatmap_needs_absolute_fallback = prepared => {
    if (!percent_requested()) return false;
    const heatmap = (prepared.data || []).find(trace => trace.type === "heatmap");
    if (!heatmap || !Array.isArray(heatmap.customdata)) return false;
    const current_losses = Array.isArray(prepared.layout?.meta?.thog2_current_losses)
      ? prepared.layout.meta.thog2_current_losses
      : [];
    let saw_data = false;
    for (let row_index = 0; row_index < heatmap.customdata.length; row_index += 1) {
      const row = heatmap.customdata[row_index];
      if (!Array.isArray(row)) continue;
      const row_has_delta = row.some(cell => (
        Array.isArray(cell) && finite_number(cell[3]) !== null
      ));
      if (!row_has_delta) continue;
      saw_data = true;
      const current_loss = finite_number(current_losses[row_index]);
      if (current_loss === null || current_loss === 0) return true;
    }
    return saw_data && current_losses.length === 0;
  };

  const apply_legacy_absolute_heatmap = prepared => {
    const heatmap = (prepared.data || []).find(trace => trace.type === "heatmap");
    if (!heatmap || !Array.isArray(heatmap.customdata)) return false;
    const raw_values = [];
    for (const row of heatmap.customdata) {
      if (!Array.isArray(row)) continue;
      for (const cell of row) {
        if (!Array.isArray(cell)) continue;
        const raw_delta = finite_number(cell[3]);
        if (raw_delta === null) continue;
        cell[5] = raw_delta;
        raw_values.push(raw_delta);
      }
    }
    const settings = heatmap_settings_for_current_run() || {};
    const limits = settings.auto_colour_saturation === true
      ? limits_from_values(raw_values)
      : manual_band_limits();
    heatmap.z = heatmap.customdata.map(row => (
      Array.isArray(row)
        ? row.map(cell => {
            const raw_delta = Array.isArray(cell) ? finite_number(cell[3]) : null;
            return raw_delta === null ? null : band_value(raw_delta, limits);
          })
        : row
    ));
    heatmap.zmin = -1;
    heatmap.zmax = 1;
    heatmap.zmid = 0;
    heatmap.colorscale = [
      [0.000, "rgb(255,226,0)"],
      [0.120, "rgb(108,96,43)"],
      [0.130, "rgb(0,126,255)"],
      [0.245, "rgb(48,72,104)"],
      [0.255, "rgb(0,255,0)"],
      [0.495, "rgb(72,96,72)"],
      [0.500, "rgb(88,88,88)"],
      [0.505, "rgb(112,76,76)"],
      [1.000, "rgb(255,0,0)"],
    ];
    heatmap.hovertemplate = (
      "step=%{customdata[0]}<br>"
      + "layer count (abs) = %{customdata[1]}<br>"
      + "layer count (rel) = %{customdata[2]}<br>"
      + "Δloss=%{customdata[3]:.8f}<extra></extra>"
    );
    heatmap.colorbar = {
      ...(heatmap.colorbar || {}),
      tickmode: "array",
      tickvals: [-1, -0.76, -0.74, -0.51, -0.49, 0, 1],
      ticktext: [
        `yellow ${format_limit(limits.yellow, "−")}`,
        "yellow ≤ −1",
        `blue ${format_limit(limits.blue, "−")}`,
        "blue ≤ −0.1",
        `green ${format_limit(limits.green, "−")}`,
        "0",
        `red ${format_limit(limits.red, "+")}`,
      ],
      title: "Δloss bands (legacy absolute fallback)",
    };
    prepared.layout = prepared.layout || {};
    prepared.layout.meta = {
      ...(prepared.layout.meta || {}),
      thog2_legacy_absolute_fallback: true,
    };
    return true;
  };

  const sync_legacy_heatmap_button = fallback => {
    const button = by_id("heatmap_delta_loss_mode");
    if (!button) return;
    if (!fallback) {
      button.disabled = false;
      return;
    }
    button.textContent = "|abs|";
    button.dataset.mode = "absolute-legacy";
    button.disabled = true;
    button.setAttribute("aria-pressed", "false");
    button.title = "Percentage Δloss is unavailable for this legacy run because centre-loss metadata was not recorded; showing absolute Δloss.";
    button.setAttribute("aria-label", button.title);
  };

  const base_transpose_heatmap = transpose_heatmap;
  transpose_heatmap = function(prepared) {
    const result = base_transpose_heatmap(prepared);
    const fallback = heatmap_needs_absolute_fallback(prepared);
    if (fallback) apply_legacy_absolute_heatmap(prepared);
    else if (prepared.layout?.meta) delete prepared.layout.meta.thog2_legacy_absolute_fallback;
    queueMicrotask(() => sync_legacy_heatmap_button(fallback));
    return result;
  };

  window.__instra_legacy_heatmap_repair = Object.freeze({
    heatmap_needs_absolute_fallback,
    apply_legacy_absolute_heatmap,
  });
});
// ^^^ THOG
