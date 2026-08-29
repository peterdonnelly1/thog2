// vvv THOG
"use strict";

// Final local-dashboard presentation/settings pass: preserve literal heatmap row
// pitch while giving the colour key a useful minimum height, expose the Overview
// default font size in Settings, align the newest y-step label, darken pencil-line
// panel separators, and reserve sufficient axis room for coefficient charts.
window.addEventListener("load", () => {
  setTimeout(() => {
    const heatmap_chrome_height_px = 180;
    const colour_key_floor_px = 64;
    const colour_key_normal_cap_px = 220;
    const overview_default_storage_key = "thog2_local_overview_default_font_size";
    const overview_current_storage_key = "thog2_local_overview_font_size";
    const trajectory_scale_settings_key = "thog2_local_trajectory_scale_modes";
    const trajectory_chart_names = new Set([
      "attn_q_head_N",
      "attn_k_head_N",
      "attn_v_head_N",
      "attn_out_head_N",
      "mlp_up",
      "mlp_down",
    ]);

    const finite_number = value => {
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };

    const visible_heatmap_height = mount => {
      const shell = mount?.closest(".heatmap-shell") || document.querySelector(".heatmap-shell");
      return Math.max(1, Number(shell?.clientHeight || 1));
    };

    const heatmap_key_minimum_height = mount => Math.max(
      colour_key_floor_px,
      Math.round(visible_heatmap_height(mount) * 0.25),
    );

    const heatmap_required_canvas_height = (mount, figure) => {
      const probe_count = Math.max(1, heatmap_probe_count(figure));
      const body_height = probe_count * heatmap_probe_row_height_px();
      const natural_height = heatmap_chrome_height_px + body_height;
      const key_minimum = heatmap_key_minimum_height(mount);
      // Extra room accommodates the key title/ticks without changing heatmap row pitch.
      return Math.max(natural_height, key_minimum + 40);
    };

    const base_plot_mount_dimensions_presentation = plot_mount_dimensions;
    plot_mount_dimensions = function(mount, chart_name, figure) {
      const dimensions = base_plot_mount_dimensions_presentation(mount, chart_name, figure);
      if (chart_name !== "heatmap") return dimensions;
      return {
        ...dimensions,
        height: heatmap_required_canvas_height(mount, figure),
      };
    };

    const base_transpose_heatmap_presentation = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      base_transpose_heatmap_presentation(prepared);
      const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (!heatmap_trace || !prepared.layout?.yaxis) return;

      // Both native x axes receive their own fixed margin and title standoff.
      // The top title sits on a separate line above the ordinates.
      for (const axis_name of ["xaxis", "xaxis2"]) {
        const axis = prepared.layout?.[axis_name];
        if (!axis) continue;
        const title = axis.title;
        const title_text = typeof title === "string"
          ? title
          : String(title?.text || "candidate layer-count offset from active layer count");
        prepared.layout[axis_name] = {
          ...axis,
          title: {
            ...(typeof title === "object" && title !== null ? title : {}),
            text: title_text,
            standoff: axis_name === "xaxis" ? 28 : 12,
          },
          automargin: false,
        };
      }
      prepared.layout.margin = {
        ...(prepared.layout.margin || {}),
        t: 104,
        b: 76,
      };
      prepared.layout.annotations = (prepared.layout.annotations || []).filter(
        annotation => annotation?.name !== "thog2-heatmap-x-title"
      );

      const mount = by_id("heatmap_plot");
      const probe_count = Math.max(1, Array.isArray(heatmap_trace.y) ? heatmap_trace.y.length : 1);
      const body_height = probe_count * heatmap_probe_row_height_px();
      const canvas_height = heatmap_required_canvas_height(mount, app.figures?.heatmap || {});
      const margin = prepared.layout.margin || {};
      const top_margin = Math.max(0, Number(margin.t || 0));
      const bottom_margin = Math.max(0, Number(margin.b || 0));
      const plot_area_height = Math.max(1, canvas_height - top_margin - bottom_margin);

      // If the canvas was enlarged solely to accommodate a useful key, keep the
      // heatmap itself at literal px/step and anchor it to the top of the plot area.
      const domain_fraction = Math.max(0.001, Math.min(1, body_height / plot_area_height));
      prepared.layout.yaxis.domain = [1 - domain_fraction, 1];

      const key_minimum = heatmap_key_minimum_height(mount);
      const desired_key_height = Math.max(
        key_minimum,
        Math.round(body_height * 0.55),
      );
      const key_cap = Math.max(colour_key_normal_cap_px, key_minimum);
      const key_height = Math.min(desired_key_height, key_cap);
      const existing_colourbar = heatmap_trace.colorbar || {};
      const existing_colourbar_title = existing_colourbar.title;

      heatmap_trace.showscale = true;
      heatmap_trace.colorbar = {
        ...existing_colourbar,
        x: 1.025,
        xanchor: "left",
        xpad: 8,
        y: 0.5,
        yanchor: "middle",
        len: key_height,
        lenmode: "pixels",
        thickness: 13,
        thicknessmode: "pixels",
        tickfont: {
          ...(existing_colourbar.tickfont || {}),
          size: 12,
        },
        title: typeof existing_colourbar_title === "string"
          ? {text: existing_colourbar_title, side: "top", font: {size: 12}}
          : {
              ...(existing_colourbar_title || {}),
              font: {
                ...(existing_colourbar_title?.font || {}),
                size: 12,
              },
            },
      };
    };

    const trajectory_mode = chart_name => (
      load_json(trajectory_scale_settings_key, {})[chart_name] === "log" ? "log" : "linear"
    );

    const base_prepare_figure_presentation = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_presentation(figure, chart_name);
      if (chart_name === "heatmap") {
        const top_title = prepared.layout?.xaxis?.title;
        const title_object = typeof top_title === "object" && top_title !== null
          ? top_title
          : {text: String(top_title || "candidate layer-count offset from L")};
        prepared.layout.xaxis = {
          ...(prepared.layout.xaxis || {}),
          title: {...title_object, standoff: 28},
          automargin: false,
        };
        prepared.layout.xaxis2 = {
          ...(prepared.layout.xaxis2 || {}),
          title: {...title_object, standoff: 12},
          automargin: false,
        };
        prepared.layout.margin = {
          ...(prepared.layout.margin || {}),
          t: 104,
          b: 76,
        };
        return prepared;
      }
      if (!trajectory_chart_names.has(chart_name)) return prepared;

      const log_mode = trajectory_mode(chart_name) === "log";
      const margin = prepared.layout.margin || {};
      prepared.layout.margin = {
        ...margin,
        // Signed-log labels are materially wider than the linear labels.
        l: Math.max(Number(margin.l || 0), log_mode ? 92 : 68),
        // Keep layer-index title below the x tick labels even in short panels.
        b: Math.max(Number(margin.b || 0), 64),
      };
      prepared.layout.xaxis = {
        ...(prepared.layout.xaxis || {}),
        automargin: true,
      };
      const x_title = prepared.layout.xaxis.title;
      if (typeof x_title === "string") {
        prepared.layout.xaxis.title = {text: x_title, standoff: 18};
      } else {
        prepared.layout.xaxis.title = {
          ...(x_title || {}),
          standoff: 18,
        };
      }
      prepared.layout.yaxis = {
        ...(prepared.layout.yaxis || {}),
        automargin: true,
      };
      return prepared;
    };

    const install_overview_default_font_setting = () => {
      if (by_id("overview_default_font_size")) return;
      const settings_content = document.querySelector(".settings-dialog .settings-content");
      if (!settings_content) return;

      const block = document.createElement("div");
      block.className = "overview-default-font-setting";
      const label = document.createElement("label");
      label.htmlFor = "overview_default_font_size";
      label.textContent = "Overview default font size";
      const control = document.createElement("div");
      control.className = "overview-default-font-control";
      const input = document.createElement("input");
      input.id = "overview_default_font_size";
      input.type = "number";
      input.min = "8";
      input.max = "18";
      input.step = "1";
      input.value = "11";
      const unit = document.createElement("span");
      unit.textContent = "px";
      control.append(input, unit);
      block.append(label, control);
      settings_content.appendChild(block);
    };

    const overview_default_font_size = () => {
      const stored_default = finite_number(localStorage.getItem(overview_default_storage_key));
      if (stored_default !== null) return Math.max(8, Math.min(18, Math.round(stored_default)));
      const current = finite_number(localStorage.getItem(overview_current_storage_key));
      if (current !== null) return Math.max(8, Math.min(18, Math.round(current)));
      return 11;
    };

    const apply_overview_font_size_through_existing_controls = desired => {
      const target = Math.max(8, Math.min(18, Math.round(desired)));
      const pane = by_id("run_overview_pane") || document.querySelector(".run-overview-pane");
      const smaller = by_id("overview_font_smaller");
      const larger = by_id("overview_font_larger");
      let current = finite_number(
        pane ? getComputedStyle(pane).getPropertyValue("--thog2-overview-font-size") : null
      );
      if (current === null) current = finite_number(localStorage.getItem(overview_current_storage_key));
      if (current === null) current = 11;
      current = Math.max(8, Math.min(18, Math.round(current)));

      if (smaller && larger) {
        while (current < target) { larger.click(); current += 1; }
        while (current > target) { smaller.click(); current -= 1; }
      } else {
        localStorage.setItem(overview_current_storage_key, String(target));
        pane?.style.setProperty("--thog2-overview-font-size", `${target}px`);
      }
    };

    install_overview_default_font_setting();

    const base_open_settings_presentation = open_settings;
    open_settings = function() {
      install_overview_default_font_setting();
      const result = base_open_settings_presentation();
      const input = by_id("overview_default_font_size");
      if (input) input.value = String(overview_default_font_size());
      return result;
    };

    const base_save_settings_presentation = save_settings;
    save_settings = function() {
      const input = by_id("overview_default_font_size");
      const desired = finite_number(input?.value);
      if (desired === null || desired < 8 || desired > 18) {
        show_toast("Overview default font size must be between 8 and 18 px.");
        return;
      }
      // Avoid applying the Overview value if the established run-timeout field
      // is invalid and the settings dialog is about to reject the save anyway.
      const timeout = finite_number(by_id("timeout_minutes")?.value);
      if (timeout === null || timeout < 1 || timeout > 10080) {
        return base_save_settings_presentation();
      }
      const rounded = Math.round(desired);
      localStorage.setItem(overview_default_storage_key, String(rounded));
      apply_overview_font_size_through_existing_controls(rounded);
      return base_save_settings_presentation();
    };

    const mark_newest_y_tick = () => {
      const texts = [...document.querySelectorAll('#heatmap_plot .ytick text')];
      for (const text of texts) text.classList.remove("thog2-newest-y-tick");
      if (!texts.length) return;
      const newest = texts.reduce((best, candidate) => (
        candidate.getBoundingClientRect().top < best.getBoundingClientRect().top ? candidate : best
      ));
      newest.classList.add("thog2-newest-y-tick");
    };

    const base_render_plot_presentation = render_plot;
    render_plot = async function(mount, figure, chart_name) {
      const result = await base_render_plot_presentation(mount, figure, chart_name);
      if (chart_name === "heatmap") requestAnimationFrame(mark_newest_y_tick);
      return result;
    };

    const style = document.createElement("style");
    style.textContent = `
      .overview-default-font-setting {
        display: grid;
        grid-template-columns: minmax(210px, 1fr) 150px;
        gap: 12px;
        align-items: center;
        margin-top: 18px;
        padding-top: 18px;
        border-top: 1px solid var(--border);
      }
      .overview-default-font-setting > label {
        font-size: 11px;
      }
      .overview-default-font-control {
        display: flex;
        align-items: center;
        gap: 7px;
      }
      .overview-default-font-control input {
        width: 92px;
        height: 32px;
        padding: 0 8px;
        border: 1px solid var(--border);
        border-radius: 4px;
        background: #fff;
      }
      .overview-default-font-control span {
        color: var(--muted);
        font-size: 11px;
      }

      /* Moderate darker-than-before pencil separators, still lighter than mid-grey. */
      .chart-grid,
      .local-metric-group .local-metric-grid {
        background: #adb4bd !important;
      }
      .chart-card-header {
        border-bottom-color: #b6bdc6 !important;
      }
      .panel-resizer-east { border-right-color: #929ba6 !important; }
      .panel-resizer-south { border-bottom-color: #929ba6 !important; }

      /* The newest/top step should be visually centred on the first heatmap row. */
      .chart-card[data-chart="heatmap"] .ytick text.thog2-newest-y-tick {
        font-size: 17px !important;
        font-weight: 800 !important;
        fill: #171a1f !important;
        transform: translate(-7px, 10px) !important;
      }
    `;
    document.head.appendChild(style);

    // Rebuild visible charts once so the new axis margins/key geometry apply
    // immediately after a hard refresh, without waiting for the next data poll.
    requestAnimationFrame(() => requestAnimationFrame(async () => {
      const heatmap_mount = by_id("heatmap_plot");
      if (heatmap_mount && app.figures?.heatmap && app.current_run_id) {
        await render_plot(heatmap_mount, app.figures.heatmap, "heatmap");
      }
      for (const chart_name of trajectory_chart_names) {
        const mount = by_id(`${chart_name}_plot`);
        const figure = app.figures?.depth?.[chart_name];
        if (mount && figure && app.current_run_id) await render_plot(mount, figure, chart_name);
      }
    }));
  }, 220);
});
// ^^^ THOG
