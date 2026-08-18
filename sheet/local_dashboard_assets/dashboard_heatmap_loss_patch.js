// vvv THOG
"use strict";

// Final local-dashboard overlay for centre/L loss metadata, absolute/percentage
// heatmap colouring, and one-shot deletion of all ticked local runs.
window.addEventListener("load", () => {
  setTimeout(() => {
    const display_mode_setting = "delta_loss_display_mode";
    const current_display_mode = () => (
      heatmap_settings_for_current_run()[display_mode_setting] === "percent"
        ? "percent"
        : "absolute"
    );
    const clamp_01 = value => Math.max(0, Math.min(1, Number(value)));
    const finite_number = value => {
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };
    const viewer_limit = (name, fallback) => {
      const value = finite_number(heatmap_settings_for_current_run()[name]);
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
    const auto_enabled = () => heatmap_settings_for_current_run().auto_colour_saturation === true;

    const display_delta = (raw_delta, current_loss) => {
      if (current_display_mode() !== "percent") return raw_delta;
      if (!(Number.isFinite(current_loss) && current_loss !== 0)) return null;
      return 100.0 * raw_delta / current_loss;
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
        const intensity = clamp_01((Math.abs(value) - 1.0) / denominator);
        return -0.76 - 0.24 * intensity;
      }
      if (value <= -0.1) {
        const denominator = Math.max(1e-12, limits.blue - 0.1);
        const intensity = clamp_01((Math.abs(value) - 0.1) / denominator);
        return -0.51 - 0.23 * intensity;
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

    const signed_fixed_2 = value => {
      const numeric = finite_number(value);
      if (numeric === null) return "—";
      return `${numeric >= 0 ? "+" : "-"}${Math.abs(numeric).toFixed(2)}`;
    };

    const annotation_delta_colour = value => {
      if (!Number.isFinite(value) || value === 0) return "rgb(220,220,220)";
      return value < 0 ? "rgb(102,255,0)" : "rgb(255,0,0)";
    };

    const centre_annotations = (prepared, heatmap_trace, current_losses) => {
      const coordinates = Array.isArray(heatmap_trace.y) ? heatmap_trace.y : [];
      const customdata = Array.isArray(heatmap_trace.customdata) ? heatmap_trace.customdata : [];
      const row_height = heatmap_probe_row_height_px();
      const all_rows_are_legible = row_height >= 10;
      const retained_indices = new Set();
      if (all_rows_are_legible) {
        for (let index = 0; index < coordinates.length; index += 1) retained_indices.add(index);
      } else {
        const tickvals = Array.isArray(prepared.layout?.yaxis?.tickvals)
          ? prepared.layout.yaxis.tickvals
          : [];
        const coordinate_to_index = new Map(
          coordinates.map((coordinate, index) => [Number(coordinate), index])
        );
        for (const tick of tickvals) {
          const index = coordinate_to_index.get(Number(tick));
          if (index !== undefined) retained_indices.add(index);
        }
        if (coordinates.length) retained_indices.add(coordinates.length - 1);
      }

      const annotations = [];
      for (const index of [...retained_indices].sort((left, right) => left - right)) {
        const loss = finite_number(current_losses[index]);
        if (loss === null) continue;
        const previous_loss = index > 0 ? finite_number(current_losses[index - 1]) : null;
        const delta = previous_loss === null ? null : loss - previous_loss;
        const step = Array.isArray(customdata[index]?.[0])
          ? customdata[index][0][0]
          : customdata[index]?.[0]?.[0];
        const loss_text = loss.toFixed(2);
        const delta_text = delta === null ? "Δ=     —" : `Δ= ${signed_fixed_2(delta)}`;
        const common = {
          x: 0,
          y: coordinates[index],
          xref: "x",
          yref: "y",
          showarrow: false,
          yanchor: "middle",
          font: {family: "DejaVu Sans Mono, monospace", size: 10},
          captureevents: false,
          hovertext: step === undefined ? undefined : `step=${step}`,
        };
        annotations.push({
          ...common,
          text: loss_text,
          xanchor: "right",
          xshift: -4,
          font: {...common.font, color: "rgb(255,255,255)"},
        });
        annotations.push({
          ...common,
          text: delta_text,
          xanchor: "left",
          xshift: 4,
          font: {...common.font, color: annotation_delta_colour(delta)},
        });
      }
      return annotations;
    };

    const format_limit = (value, sign, percent_mode) => {
      if (!(Number(value) > 0)) return "—";
      return `${sign}${Number(value).toPrecision(3)}${percent_mode ? "%" : ""}`;
    };

    const base_transpose_heatmap_loss = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      base_transpose_heatmap_loss(prepared);
      const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (!heatmap_trace) return;
      const active_layer_trace = (prepared.data || []).find(
        trace => trace !== heatmap_trace && Array.isArray(trace.x) && Array.isArray(trace.y)
      );
      if (active_layer_trace) {
        active_layer_trace.visible = false;
        active_layer_trace.showlegend = false;
        active_layer_trace.hoverinfo = "skip";
      }

      const current_losses = Array.isArray(prepared.layout?.meta?.thog2_current_losses)
        ? prepared.layout.meta.thog2_current_losses.map(finite_number)
        : [];
      const customdata = Array.isArray(heatmap_trace.customdata) ? heatmap_trace.customdata : [];
      const display_values = [];
      for (let row_index = 0; row_index < customdata.length; row_index += 1) {
        const current_loss = current_losses[row_index] ?? null;
        const row = customdata[row_index];
        if (!Array.isArray(row)) continue;
        for (let column_index = 0; column_index < row.length; column_index += 1) {
          const cell = row[column_index];
          if (!Array.isArray(cell)) continue;
          const raw_delta = finite_number(cell[3]);
          if (raw_delta === null) continue;
          const shown_delta = display_delta(raw_delta, current_loss);
          cell[4] = current_loss;
          cell[5] = shown_delta;
          if (shown_delta !== null) display_values.push(shown_delta);
        }
      }

      const limits = auto_enabled() ? limits_from_values(display_values) : manual_band_limits();
      heatmap_trace.z = customdata.map(row => (
        Array.isArray(row)
          ? row.map(cell => {
              const shown_delta = Array.isArray(cell) ? finite_number(cell[5]) : null;
              return shown_delta === null ? null : band_value(shown_delta, limits);
            })
          : row
      ));
      heatmap_trace.zmin = -1;
      heatmap_trace.zmax = 1;
      heatmap_trace.zmid = 0;
      heatmap_trace.colorscale = [
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
      const percent_mode = current_display_mode() === "percent";
      heatmap_trace.hovertemplate = percent_mode
        ? (
            "step=%{customdata[0]}<br>"
            + "layer count (abs) = %{customdata[1]}<br>"
            + "layer count (rel) = %{customdata[2]}<br>"
            + "Δloss=%{customdata[3]:.8f}<br>"
            + "Δloss (%)=%{customdata[5]:+.5f}%<extra></extra>"
          )
        : (
            "step=%{customdata[0]}<br>"
            + "layer count (abs) = %{customdata[1]}<br>"
            + "layer count (rel) = %{customdata[2]}<br>"
            + "Δloss=%{customdata[3]:.8f}<extra></extra>"
          );
      heatmap_trace.colorbar = {
        ...(heatmap_trace.colorbar || {}),
        tickmode: "array",
        tickvals: [-1, -0.76, -0.74, -0.51, -0.49, 0, 1],
        ticktext: [
          `${auto_enabled() ? "auto " : ""}yellow ${format_limit(limits.yellow, "−", percent_mode)}`,
          `yellow ≤ −1${percent_mode ? "%" : ""}`,
          `${auto_enabled() ? "auto " : ""}blue ${format_limit(limits.blue, "−", percent_mode)}`,
          `blue ≤ −0.1${percent_mode ? "%" : ""}`,
          `${auto_enabled() ? "auto " : ""}green ${format_limit(limits.green, "−", percent_mode)}`,
          "0",
          `${auto_enabled() ? "auto " : ""}red ${format_limit(limits.red, "+", percent_mode)}`,
        ],
        title: percent_mode ? "Δloss (%) bands" : "Δloss bands",
      };

      queueMicrotask(sync_heatmap_loss_mode_button);
    };

    function sync_heatmap_loss_mode_button() {
      const button = by_id("heatmap_delta_loss_mode");
      if (!button) return;
      const percent_mode = current_display_mode() === "percent";
      button.textContent = percent_mode ? "%" : "Abs";
      button.dataset.mode = percent_mode ? "percent" : "absolute";
      button.title = percent_mode
        ? "Heatmap colours use percentage Δloss relative to the centre L loss; click for absolute Δloss"
        : "Heatmap colours use absolute Δloss; click for percentage Δloss relative to the centre L loss";
      button.setAttribute("aria-label", button.title);
      button.setAttribute("aria-pressed", String(percent_mode));
      button.disabled = !app.current_run_id;
    }

    const heatmap_header = document.querySelector('.chart-card[data-chart="heatmap"] .chart-card-header');
    const vertical_control = by_id("heatmap_vertical_scale")?.closest(".heatmap-vertical-scale-control");
    const maximize = heatmap_header?.querySelector(".maximize-button");
    const actions = maximize?.parentElement;
    if (heatmap_header && actions && !by_id("heatmap_delta_loss_mode")) {
      const button = document.createElement("button");
      button.id = "heatmap_delta_loss_mode";
      button.type = "button";
      button.className = "heatmap-delta-loss-mode";
      button.addEventListener("click", async event => {
        event.preventDefault();
        event.stopPropagation();
        const next = current_display_mode() === "absolute" ? "percent" : "absolute";
        save_heatmap_viewer_setting(display_mode_setting, next);
        sync_heatmap_loss_mode_button();
        if (app.figures && app.current_run_id) await render_figures();
      });
      actions.insertBefore(button, vertical_control || maximize);
    }
    sync_heatmap_loss_mode_button();

    const style = document.createElement("style");
    style.textContent = `
      .heatmap-delta-loss-mode,
      .bulk-delete-runs-button {
        height: 29px;
        min-width: 38px;
        border: 1px solid #cfd3d8;
        border-radius: 5px;
        background: #f6f7f8;
        color: #505862;
        font: inherit;
        font-size: 11px;
        cursor: pointer;
      }
      .heatmap-delta-loss-mode:hover:not(:disabled),
      .bulk-delete-runs-button:hover:not(:disabled) { background: #eef0f2; }
      .heatmap-delta-loss-mode[data-mode="percent"] {
        border-color: #a8a0df;
        background: #efecff;
        color: #5140b7;
        font-weight: 700;
      }
      .bulk-delete-runs-button {
        width: 32px;
        min-width: 32px;
        padding: 5px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
      }
      .bulk-delete-runs-button svg {
        width: 15px;
        height: 15px;
        fill: none;
        stroke: currentColor;
        stroke-width: 1.7;
        stroke-linecap: round;
        stroke-linejoin: round;
      }
      .bulk-delete-runs-button:disabled,
      .heatmap-delta-loss-mode:disabled { opacity: 0.42; cursor: default; }
    `;
    document.head.appendChild(style);

    const reset_current_run_after_bulk_delete = () => {
      if (app.maximized_chart) restore_maximized_chart();
      app.current_run_id = null;
      app.current_status = null;
      app.figures = null;
      app.figure_revision = null;
      app.manual_selection = false;
      by_id("heatmap_plot")?.replaceChildren();
      if (by_id("heatmap_plot")) by_id("heatmap_plot").dataset.plotReady = "false";
      history.replaceState({}, "", "/runs");
    };

    const toolbar = document.querySelector(".runs-pane-header .toolbar");
    let bulk_delete = by_id("delete_selected_runs");
    if (toolbar && !bulk_delete) {
      bulk_delete = document.createElement("button");
      bulk_delete.id = "delete_selected_runs";
      bulk_delete.type = "button";
      bulk_delete.className = "bulk-delete-runs-button";
      bulk_delete.title = "Delete all ticked local runs";
      bulk_delete.setAttribute("aria-label", bulk_delete.title);
      bulk_delete.innerHTML = (
        '<svg viewBox="0 0 20 20" aria-hidden="true">'
        + '<path d="M3.5 5.5h13"/><path d="M7 5.5V3.7h6v1.8"/>'
        + '<path d="M5.6 5.5l.8 11h7.2l.8-11"/><path d="M8.3 8.2v5.6M11.7 8.2v5.6"/>'
        + '</svg>'
      );
      toolbar.prepend(bulk_delete);
    }

    const sync_bulk_delete = () => {
      if (!bulk_delete) return;
      const selected_ids = [...app.selected].filter(run_id => (
        app.runs.some(run => run_identifier(run) === run_id)
      ));
      bulk_delete.disabled = selected_ids.length === 0;
      bulk_delete.title = selected_ids.length
        ? `Delete ${selected_ids.length} ticked local run${selected_ids.length === 1 ? "" : "s"}`
        : "Delete all ticked local runs";
      bulk_delete.setAttribute("aria-label", bulk_delete.title);
    };

    bulk_delete?.addEventListener("click", async () => {
      const selected_ids = [...app.selected].filter(run_id => (
        app.runs.some(run => run_identifier(run) === run_id)
      ));
      if (!selected_ids.length) return;
      const selected_runs = selected_ids
        .map(run_id => app.runs.find(run => run_identifier(run) === run_id))
        .filter(Boolean);
      const running_count = selected_runs.filter(run => display_run_state(run) === "running").length;
      const warning = running_count
        ? `\n\n${running_count} selected run${running_count === 1 ? " is" : "s are"} still marked running.`
        : "";
      if (!window.confirm(
        `Delete ${selected_ids.length} ticked local run${selected_ids.length === 1 ? "" : "s"}?`
        + "\n\nThis removes their local chart databases only."
        + warning
      )) return;

      bulk_delete.disabled = true;
      const deleted = [];
      const failed = [];
      for (const run_id of selected_ids) {
        try {
          await fetch_json(`/api/run?run=${encodeURIComponent(run_id)}`, {method: "DELETE"});
          deleted.push(run_id);
          app.selected.delete(run_id);
          delete app.colours[run_id];
          delete app.visibility[run_id];
        } catch (error) {
          failed.push([run_id, error]);
        }
      }
      save_json("thog2_local_run_colours", app.colours);
      save_json("thog2_local_run_visibility", app.visibility);
      if (deleted.includes(app.current_run_id)) reset_current_run_after_bulk_delete();
      await refresh_catalog();
      sync_bulk_delete();
      if (failed.length) {
        show_toast(`Deleted ${deleted.length}; failed to delete ${failed.length}.`);
      } else {
        show_toast(`Deleted ${deleted.length} local run${deleted.length === 1 ? "" : "s"}.`);
      }
    });

    by_id("runs_body")?.addEventListener("change", event => {
      if (event.target instanceof HTMLInputElement && event.target.type === "checkbox") {
        queueMicrotask(sync_bulk_delete);
      }
    });
    by_id("select_all")?.addEventListener("change", () => queueMicrotask(sync_bulk_delete));
    const base_render_runs_loss_patch = render_runs;
    render_runs = function() {
      base_render_runs_loss_patch();
      sync_bulk_delete();
    };
    sync_bulk_delete();

    // A newly selected run can receive its figure before the heatmap viewport has its final
    // width/height. Force one post-layout resize so the stale/squashed first render does not
    // require switching away and back to repair it.
    const base_select_run_loss_patch = select_run;
    select_run = async function(...args) {
      const result = await base_select_run_loss_patch(...args);
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const card = document.querySelector('.chart-card[data-chart="heatmap"]');
        if (card && card.offsetParent !== null) resize_plot_in_card(card);
      }));
      return result;
    };

    if (app.figures && app.current_run_id) render_figures();
  }, 1);
});
// ^^^ THOG
