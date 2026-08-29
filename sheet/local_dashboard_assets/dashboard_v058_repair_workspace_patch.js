// vvv THOG
"use strict";

// One final owner for the v0.58 Instra presentation and multi-run Workspace.
// Earlier files remain compatibility layers for stored settings and old runs;
// this layer deliberately resolves their conflicting late axis/annotation rules.
window.addEventListener("load", () => {
  let pending_workspace_enter = false;
  const early_workspace_nav = by_id("workspace_nav");
  const early_runs_nav = by_id("runs_nav");
  const queue_workspace_enter = () => { pending_workspace_enter = true; };
  const cancel_workspace_enter = () => { pending_workspace_enter = false; };
  early_workspace_nav?.addEventListener("click", queue_workspace_enter);
  early_runs_nav?.addEventListener("click", cancel_workspace_enter);

  setTimeout(() => {
    const weight_chart_names = Object.freeze([
      "attn_q_head_N",
      "attn_k_head_N",
      "attn_v_head_N",
      "attn_out_head_N",
      "mlp_up",
      "mlp_down",
    ]);
    const weight_chart_set = new Set(weight_chart_names);
    const heatmap_title = "Heatmap - True Loss vs Counterfactual Layer Count Loss";
    const old_heatmap_titles = new Set([
      "Layer-count Δloss heatmap",
      "Heatmap - Loss vs Counterfactual Layer Count",
    ]);
    const weight_scale_key = "thog2_local_trajectory_scale_modes";
    const salmon = "#ff9696";
    const pale_blue = "#96dcff";
    let workspace_refresh_timer = null;
    let pending_heatmap_settings = null;
    let last_workspace_selection_key = "";

    const finite_number = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };
    const clone = value => JSON.parse(JSON.stringify(value));
    const run_name = run => String(run?.artifact_name || run?.run_name || run_identifier(run));
    const run_datetime = run => {
      const artifact = run_name(run);
      const match = artifact.match(/^(\d{6}-\d{4}|\d{2}-\d{3,4}-\d{4})(?:_|$)/);
      return match ? match[1] : artifact;
    };
    const escaped_html = value => String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
    const source_optimizer_update = trace => {
      for (const value of [
        trace?.meta?.instra_workspace_optimizer_update,
        trace?.meta?.instra_dense_optimizer_update,
        trace?.meta?.instra_thog_optimizer_update,
      ]) {
        const numeric = finite_number(value);
        if (numeric !== null) return numeric;
      }
      const description = `${trace?.name || ""} ${trace?.hovertemplate || ""}`;
      const step_match = description.match(/(?:^|[^A-Za-z0-9])step\s+(\d+)(?:\D|$)/i);
      if (step_match) return Number(step_match[1]);
      const update_match = description.match(/(?:^|[^A-Za-z0-9])U(\d+)(?:\D|$)/);
      return update_match ? Number(update_match[1]) : null;
    };
    const visible_runs = () => app.runs.filter(run => is_visible(run_identifier(run)));
    const dense_run = run => String(run?.model_type || "").trim().toLowerCase() === "dense";
    const heatmap_available = run => {
      const configuration = run?.configuration && typeof run.configuration === "object"
        ? run.configuration
        : {};
      return (
        Number(run?.heatmap_count || 0) > 0
        || run?.heatmap_settings?.mode === true
        || configuration.instrumentation__delta_loss_v_layer_heatmap === true
      );
    };
    const direct_json = async url => {
      const response = await fetch(url, {cache: "no-store"});
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || `${response.status} ${response.statusText}`);
      return value;
    };
    const map_with_concurrency = async (values, limit, operation) => {
      const output = new Array(values.length);
      let cursor = 0;
      const worker = async () => {
        while (cursor < values.length) {
          const index = cursor;
          cursor += 1;
          try {
            output[index] = await operation(values[index], index);
          } catch (_error) {
            output[index] = null;
          }
        }
      };
      await Promise.all(Array.from({length: Math.min(limit, values.length)}, worker));
      return output;
    };
    const selection_key = () => visible_runs().map(run => {
      const id = run_identifier(run);
      return `${id}:${JSON.stringify(run.revision || [])}:${colour_for_run(id)}`;
    }).join("|");

    const dense_step_colour = optimizer_update => {
      const update = Math.max(0, Math.trunc(Number(optimizer_update) || 0));
      const seed = Math.imul(update, 0x9e3779b1) >>> 0;
      const hue = seed % 360;
      const saturation = 62 + ((seed >>> 8) % 17);
      const lightness = 43 + ((seed >>> 16) % 12);
      return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    };

    const merge_depth_payloads = entries => {
      const depth = {};
      for (const chart_name of weight_chart_names) {
        let merged = null;
        for (const entry of entries) {
          const source = entry?.payload?.depth?.[chart_name];
          if (!source) continue;
          if (!merged) {
            merged = clone(source);
            merged.data = [];
            merged.layout = clone(source.layout || {});
          }
          const id = run_identifier(entry.run);
          const colour = colour_for_run(id);
          for (const source_trace of source.data || []) {
            const trace = clone(source_trace);
            const prior_meta = trace.meta && typeof trace.meta === "object" && !Array.isArray(trace.meta)
              ? trace.meta
              : {};
            const dense_trace = prior_meta.instra_dense_weight === true;
            const optimizer_update = source_optimizer_update(trace);
            const dense_update = optimizer_update === null ? "—" : optimizer_update;
            const owner_name = run_name(entry.run);
            trace.meta = {
              ...prior_meta,
              instra_workspace_run_id: id,
              instra_workspace_colour: colour,
              instra_workspace_optimizer_update: optimizer_update,
              instra_workspace_artifact_name: owner_name,
              instra_workspace_run_datetime: run_datetime(entry.run),
            };
            trace.hovertemplate = `<b>${escaped_html(owner_name)}</b><br>${String(trace.hovertemplate || "")}`;
            if (dense_trace) {
              trace.name = `${owner_name} · step ${dense_update}`;
              trace.legendgroup = `instra-workspace-${id}-step-${dense_update}`;
              trace.showlegend = false;
            } else {
              trace.name = `${owner_name} · ${String(trace.name || chart_titles[chart_name])}`;
              trace.legendgroup = `instra-workspace-${id}`;
              trace.showlegend = false;
            }
            merged.data.push(trace);
          }
        }
        if (merged) {
          merged.layout = merged.layout || {};
          merged.layout.showlegend = false;
          delete merged.layout.legend;
          depth[chart_name] = merged;
        }
      }
      return {depth};
    };

    const fetch_depth_payload = async request => {
      const runs = visible_runs().filter(run => Number(run.depth_snapshot_count || 0) > 0);
      const entries = await map_with_concurrency(runs, 8, async run => ({
        run,
        payload: await request(
          `/api/figure-family?run=${encodeURIComponent(run_identifier(run))}&family=depth`
        ),
      }));
      return merge_depth_payloads(entries.filter(Boolean));
    };

    const fetch_metric_groups = async () => {
      const runs = visible_runs();
      const entries = await map_with_concurrency(runs, 8, async run => ({
        run,
        payload: await direct_json(`/api/chart-groups?run=${encodeURIComponent(run_identifier(run))}`),
      }));
      const groups = new Map();
      for (let run_index = 0; run_index < entries.length; run_index += 1) {
        const entry = entries[run_index];
        if (!entry?.payload?.available) continue;
        for (const summary of entry.payload.groups || []) {
          if (summary.name === "depth") continue;
          const current = groups.get(summary.name) || {
            name: summary.name,
            chart_count: 0,
            revision: 0,
          };
          current.chart_count = Math.max(current.chart_count, Number(summary.chart_count || 0));
          current.revision += (run_index + 1) * Number(summary.revision || 0);
          groups.set(summary.name, current);
        }
      }
      return {
        available: true,
        source: "visible Instra runs",
        groups: [...groups.values()],
      };
    };

    const intersect_modes = (left, right) => {
      if (left === null) return [...right];
      const wanted = new Set(right);
      return left.filter(value => wanted.has(value));
    };

    const fetch_metric_group = async group_name => {
      const runs = visible_runs();
      const entries = await map_with_concurrency(runs, 8, async run => ({
        run,
        payload: await direct_json(
          `/api/chart-group?run=${encodeURIComponent(run_identifier(run))}`
          + `&group=${encodeURIComponent(group_name)}`
        ),
      }));
      const charts = new Map();
      let revision = 0;
      for (let run_index = 0; run_index < entries.length; run_index += 1) {
        const entry = entries[run_index];
        const group = entry?.payload?.group;
        if (!entry?.payload?.available || !group) continue;
        revision += (run_index + 1) * Number(group.revision || 0);
        const id = run_identifier(entry.run);
        const colour = colour_for_run(id);
        for (const chart of group.charts || []) {
          let merged = charts.get(chart.id);
          if (!merged) {
            merged = {
              id: chart.id,
              title: chart.title,
              x_title: chart.x_title,
              default_x_axis_mode: chart.default_x_axis_mode,
              available_x_axis_modes: null,
              series: [],
            };
            charts.set(chart.id, merged);
          }
          merged.available_x_axis_modes = intersect_modes(
            merged.available_x_axis_modes,
            chart.available_x_axis_modes || [],
          );
          const multiple_series = (chart.series || []).length > 1;
          for (const series of chart.series || []) {
            merged.series.push({
              ...clone(series),
              name: multiple_series
                ? `${run_name(entry.run)} · ${String(series.name || chart.title)}`
                : run_name(entry.run),
              color: colour,
              instra_workspace_run_id: id,
            });
          }
        }
      }
      for (const chart of charts.values()) {
        chart.available_x_axis_modes = chart.available_x_axis_modes || [];
        if (!chart.available_x_axis_modes.includes(chart.default_x_axis_mode)) {
          chart.default_x_axis_mode = chart.available_x_axis_modes[0] || null;
        }
      }
      return {
        available: true,
        source: "visible Instra runs",
        group: {name: group_name, revision, charts: [...charts.values()]},
      };
    };

    app.workspace_mode = false;
    window.__instra_workspace = {
      active: () => app.workspace_mode === true,
      selection_key: () => `workspace:${selection_key()}`,
      visible_runs,
      fetch_depth_payload,
      fetch_metric_groups,
      fetch_metric_group,
    };

    const set_heatmap_group_visibility = () => {
      const group = by_id("heatmap_chart_group");
      if (!group) return;
      const run = current_run();
      const hide = app.workspace_mode || (dense_run(run) && !heatmap_available(run));
      group.hidden = hide;
      group.setAttribute("aria-hidden", String(hide));
      if (hide && app.maximized_chart === "heatmap") restore_maximized_chart();
    };

    const render_workspace_heading = () => {
      if (!app.workspace_mode) return;
      const runs = visible_runs();
      by_id("run_title").textContent = "Workspace";
      by_id("breadcrumb_leaf").textContent = "Workspace";
      by_id("selected_run_mark").style.background = "linear-gradient(180deg,#24abc2,#6c5bd6)";
      const subtitle = by_id("run_subtitle");
      subtitle.replaceChildren();
      const summary = document.createElement("span");
      summary.className = "identity";
      summary.textContent = `${runs.length} visible run${runs.length === 1 ? "" : "s"} · overlaid on shared axes`;
      subtitle.appendChild(summary);
      by_id("wandb_link").hidden = true;
    };

    const request_workspace_refresh = () => {
      if (!app.workspace_mode) return;
      clearTimeout(workspace_refresh_timer);
      workspace_refresh_timer = setTimeout(() => {
        const performance = window.__thog2_dashboard_performance?.state;
        if (performance) {
          performance.depth_signature = null;
          performance.pending_render = null;
          performance.deferred_coefficients = true;
        }
        app.figure_revision = null;
        if (app.figures) app.figures.depth = {};
        window.__thog2_metric_groups?.clear?.();
        window.__thog2_metric_groups?.refresh?.();
        const retry = () => {
          if (!app.workspace_mode) return;
          if (app.refresh_in_flight) {
            setTimeout(retry, 80);
            return;
          }
          refresh_current_run();
        };
        retry();
      }, 40);
    };

    const enter_workspace = () => {
      if (!app.current_run_id) {
        const first = visible_runs()[0] || app.runs[0];
        if (first) select_run(run_identifier(first), {manual: false, replace_history: true});
      }
      app.workspace_mode = true;
      document.body.classList.add("instra-workspace-mode");
      by_id("workspace_nav")?.classList.add("selected");
      by_id("runs_nav")?.classList.remove("selected");
      by_id("settings_nav")?.classList.remove("selected");
      if (typeof local_set_detail_tab === "function") local_set_detail_tab("charts");
      set_heatmap_group_visibility();
      render_workspace_heading();
      last_workspace_selection_key = selection_key();
      request_workspace_refresh();
    };

    const leave_workspace = () => {
      if (!app.workspace_mode) return;
      app.workspace_mode = false;
      document.body.classList.remove("instra-workspace-mode");
      by_id("workspace_nav")?.classList.remove("selected");
      by_id("runs_nav")?.classList.add("selected");
      set_heatmap_group_visibility();
      window.__thog2_metric_groups?.clear?.();
      app.figures = null;
      app.figure_revision = null;
      reset_run_charts();
      render_run_heading();
      refresh_current_run();
    };

    early_workspace_nav?.removeEventListener("click", queue_workspace_enter);
    early_runs_nav?.removeEventListener("click", cancel_workspace_enter);
    by_id("workspace_nav")?.addEventListener("click", enter_workspace);
    by_id("runs_nav")?.addEventListener("click", leave_workspace);
    if (pending_workspace_enter) {
      pending_workspace_enter = false;
      enter_workspace();
    }

    const base_select_run_v058 = select_run;
    select_run = function(run_id, options = {}) {
      if (app.workspace_mode && options.manual === true) leave_workspace();
      const result = base_select_run_v058(run_id, options);
      set_heatmap_group_visibility();
      return result;
    };

    const base_render_run_heading_v058 = render_run_heading;
    render_run_heading = function() {
      base_render_run_heading_v058();
      set_heatmap_group_visibility();
      render_workspace_heading();
    };

    const base_render_runs_v058 = render_runs;
    render_runs = function() {
      const result = base_render_runs_v058();
      if (app.workspace_mode) {
        const next = selection_key();
        if (next !== last_workspace_selection_key) {
          last_workspace_selection_key = next;
          request_workspace_refresh();
        }
        render_workspace_heading();
      }
      return result;
    };
    // Colour edits update run rows without calling render_runs(). Poll the cheap
    // selection signature as well so a Workspace recolours all overlaid traces.
    setInterval(() => {
      if (!app.workspace_mode) return;
      const next = selection_key();
      if (next === last_workspace_selection_key) return;
      last_workspace_selection_key = next;
      request_workspace_refresh();
    }, 500);

    const weight_mode = chart_name => (
      load_json(weight_scale_key, {})[chart_name] === "log" ? "log" : "linear"
    );
    const save_weight_mode = (chart_name, mode) => {
      const modes = load_json(weight_scale_key, {});
      modes[chart_name] = mode;
      save_json(weight_scale_key, modes);
    };

    const make_scale_button = (label, mode, title) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "explicit-mode-button";
      button.dataset.mode = mode;
      button.textContent = label;
      button.title = title;
      button.setAttribute("aria-label", title);
      return button;
    };

    const sync_weight_scale_controls = () => {
      for (const group of document.querySelectorAll(".explicit-trajectory-modes")) {
        const mode = weight_mode(group.dataset.chartName);
        for (const button of group.querySelectorAll(".explicit-mode-button")) {
          const active = button.dataset.mode === mode;
          button.classList.toggle("active", active);
          button.setAttribute("aria-pressed", String(active));
        }
      }
    };

    const install_weight_scale_controls = () => {
      for (const chart_name of weight_chart_names) {
        const card = document.querySelector(`.chart-card[data-chart="${chart_name}"]`);
        const actions = card?.querySelector(".chart-card-actions");
        if (!actions || card.querySelector(".explicit-trajectory-modes")) continue;
        card.querySelector(".trajectory-scale-toggle")?.remove();
        const group = document.createElement("div");
        group.className = "explicit-mode-group explicit-trajectory-modes";
        group.dataset.chartName = chart_name;
        group.setAttribute("role", "group");
        group.setAttribute("aria-label", `${chart_titles[chart_name]} Y scale`);
        const linear = make_scale_button("Linear", "linear", "Use a linear Y scale");
        const logarithmic = make_scale_button(
          "Log",
          "log",
          "Use the signed-log Y scale; negative and zero values remain visible",
        );
        const choose = async mode => {
          if (weight_mode(chart_name) === mode) return;
          save_weight_mode(chart_name, mode);
          sync_weight_scale_controls();
          const figure = app.figures?.depth?.[chart_name];
          const mount = by_id(`${chart_name}_plot`);
          if (figure && mount) await render_plot(mount, figure, chart_name);
        };
        linear.addEventListener("click", event => {
          event.preventDefault(); event.stopPropagation(); choose("linear");
        });
        logarithmic.addEventListener("click", event => {
          event.preventDefault(); event.stopPropagation(); choose("log");
        });
        group.append(linear, logarithmic);
        const gear = actions.querySelector(".chart-settings-button");
        actions.insertBefore(group, gear || actions.firstChild);
      }
      sync_weight_scale_controls();
    };
    install_weight_scale_controls();

    const axis_title = (value, fallback) => {
      if (typeof value === "string") return {text: value};
      return {...(value || {}), text: String(value?.text || fallback)};
    };

    // Plotly does not reliably materialise an overlaid axis unless at least one
    // trace explicitly refers to it.  Keep a transparent, non-interactive trace
    // on x2 so the mirrored ordinates and their title survive Plotly.react(),
    // settings previews, maximisation, and scroll-canvas rerenders.
    const top_axis_anchor = (prepared, x_values, y_values) => {
      prepared.data = (prepared.data || []).filter(
        trace => trace?.meta?.instra_top_axis_anchor !== true
      );
      const xs = (x_values || []).filter(value => value !== null && value !== undefined);
      const ys = (y_values || []).filter(value => value !== null && value !== undefined);
      if (!xs.length || !ys.length) return;
      const anchor_x = xs.length === 1 ? [xs[0]] : [xs[0], xs[xs.length - 1]];
      const anchor_y = anchor_x.map(() => ys[0]);
      prepared.data.push({
        type: "scatter",
        mode: "markers",
        x: anchor_x,
        y: anchor_y,
        xaxis: "x2",
        yaxis: "y",
        marker: {size: 0.1, opacity: 0},
        opacity: 0,
        showlegend: false,
        hoverinfo: "skip",
        hovertemplate: null,
        meta: {instra_top_axis_anchor: true},
      });
    };

    const base_prepare_figure_v058 = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_v058(figure, chart_name);
      if (weight_chart_set.has(chart_name)) {
        delete prepared.layout.title;
        const bottom_axis = prepared.layout.xaxis || {};
        const title = axis_title(bottom_axis.title, "layer index");
        prepared.layout.xaxis = {
          ...bottom_axis,
          side: "bottom",
          title: {...title, standoff: 16},
          automargin: true,
        };
        prepared.layout.xaxis2 = {
          ...bottom_axis,
          side: "top",
          overlaying: "x",
          matches: "x",
          visible: true,
          showticklabels: true,
          showline: true,
          ticks: "outside",
          showgrid: false,
          zeroline: false,
          automargin: true,
        };
        delete prepared.layout.xaxis2.title;
        prepared.layout.margin = {
          ...(prepared.layout.margin || {}),
          t: Math.max(42, Number(prepared.layout?.margin?.t || 0)),
          b: Math.max(64, Number(prepared.layout?.margin?.b || 0)),
        };
        const gradient_enabled = (
          window.__instra_weight_stability_final?.gradient_enabled?.() === true
        );
        for (const trace of prepared.data || []) {
          const meta = trace.meta && typeof trace.meta === "object" && !Array.isArray(trace.meta)
            ? trace.meta
            : {};
          const run_id = meta.instra_workspace_run_id;
          const dense_trace = meta.instra_dense_weight === true;
          const colour = gradient_enabled
            ? null
            : dense_trace
            ? dense_step_colour(meta.instra_dense_optimizer_update)
            : run_id
            ? colour_for_run(run_id)
            : meta.instra_workspace_colour;
          if (colour) {
            trace.line = {...(trace.line || {}), color: colour};
            trace.marker = {...(trace.marker || {}), color: colour};
            trace.marker.line = {...(trace.marker.line || {}), color: colour};
          }
          if (dense_trace && trace.marker?.symbol === "x") {
            const marker_size = finite_number(trace.marker?.size);
            trace.marker = {
              ...(trace.marker || {}),
              size: marker_size === null ? 5 : Math.min(6, marker_size),
              line: {
                ...(trace.marker.line || {}),
                width: 0.35,
                color: colour || trace.marker?.line?.color || trace.line?.color || trace.marker?.color,
              },
            };
            trace.mode = "lines+markers";
            trace.line = {
              ...(trace.line || {}),
              color: colour || trace.line?.color || trace.marker?.color,
              width: 0.45,
              shape: "linear",
            };
          }
        }
        const source_trace = (prepared.data || []).find(trace => (
          Array.isArray(trace?.x) && trace.x.length && Array.isArray(trace?.y) && trace.y.length
        ));
        top_axis_anchor(prepared, source_trace?.x, source_trace?.y);
        const presentation = window.__instra_weight_presentation;
        if (typeof presentation?.apply_plot_chrome === "function") {
          presentation.apply_plot_chrome(prepared, chart_name);
        } else {
          prepared.layout.showlegend = false;
          delete prepared.layout.legend;
          for (const trace of prepared.data || []) trace.showlegend = false;
        }
      } else if (chart_name === "heatmap") {
        const heatmap = (prepared.data || []).find(trace => trace?.type === "heatmap");
        const meta = prepared.layout?.meta || {};
        const latest_index = latest_meta_index(meta);
        const latest_l = finite_number(meta.thog2_active_layers?.[latest_index]);
        prepared.layout.xaxis2 = {
          ...(prepared.layout.xaxis2 || {}),
          side: "top",
          anchor: "y",
          overlaying: "x",
          matches: "x",
          visible: true,
          showticklabels: true,
          showline: true,
          ticks: "outside",
          showgrid: false,
          zeroline: false,
          title: {
            text: latest_l === null
              ? "absolute candidate layer count"
              : `absolute candidate layer count · latest L=${latest_l}`,
            standoff: 12,
            font: {size: 14},
          },
        };
        top_axis_anchor(prepared, heatmap?.x, heatmap?.y);
      }
      return prepared;
    };

    const heatmap_centre_font = prepared => {
      const centre = (prepared.layout?.annotations || []).find(annotation => (
        annotation?.xref === "x"
        && annotation?.yref === "y"
        && typeof annotation?.hovertext === "string"
        && annotation.hovertext.startsWith("step=")
        && String(annotation?.font?.family || "").includes("Mono")
      ));
      return {
        family: String(centre?.font?.family || "DejaVu Sans Mono, monospace"),
        // Winner and committed-decision labels deliberately inherit the exact
        // L-column typography. Do not impose an independent minimum here: the
        // centre text may shrink to fit a narrow brick.
        size: Math.max(1, Number(centre?.font?.size || 10)),
      };
    };

    const signed_offset = value => {
      const numeric = Number(value);
      if (!Number.isFinite(numeric) || numeric === 0) return "0";
      return numeric > 0 ? `+${numeric}` : String(numeric);
    };

    const latest_meta_index = meta => {
      const steps = Array.isArray(meta.thog2_optimizer_updates)
        ? meta.thog2_optimizer_updates.map(Number)
        : [];
      let index = steps.length - 1;
      let latest = -Infinity;
      for (let candidate = 0; candidate < steps.length; candidate += 1) {
        if (Number.isFinite(steps[candidate]) && steps[candidate] >= latest) {
          latest = steps[candidate];
          index = candidate;
        }
      }
      return Math.max(0, index);
    };

    const heatmap_cell_label = (loss, current_loss) => {
      if (!Number.isFinite(loss)) return "<b>decision</b>";
      const improvement = Number.isFinite(current_loss) && current_loss !== 0
        ? 100 * (current_loss - loss) / Math.abs(current_loss)
        : null;
      const suffix = Number.isFinite(improvement) ? ` (${improvement.toFixed(2)}%)` : "";
      return `<b>${loss.toFixed(3)}${suffix}</b>`;
    };

    const repair_heatmap_overlays = (prepared, heatmap) => {
      const meta = prepared.layout?.meta || {};
      const active = Array.isArray(meta.thog2_active_layers) ? meta.thog2_active_layers : [];
      const selected = Array.isArray(meta.thog2_selected_layers) ? meta.thog2_selected_layers : [];
      const current = Array.isArray(meta.thog2_current_losses) ? meta.thog2_current_losses : [];
      const decisions = Array.isArray(meta.thog2_decision_committed) ? meta.thog2_decision_committed : [];
      const offsets = Array.isArray(heatmap.x) ? heatmap.x.map(Number) : [];
      const rows = Array.isArray(heatmap.y) ? heatmap.y.map(Number) : [];
      const customdata = Array.isArray(heatmap.customdata) ? heatmap.customdata : [];
      const font = heatmap_centre_font(prepared);
      const annotations = (prepared.layout.annotations || []).filter(annotation => ![
        "thog2-best-better-loss",
        "thog2-committed-decision-text",
        "thog2-update-brake",
        "thog2-chaos-bump",
      ].includes(annotation?.name));
      const shapes = (prepared.layout.shapes || []).filter(shape => ![
        "thog2-committed-decision-brick",
        "thog2-centre-datum-background",
      ].includes(shape?.name));
      const finite_rows = rows.filter(Number.isFinite);
      const sorted_rows = [...new Set(finite_rows)].sort((left, right) => left - right);
      let pitch = 1;
      if (sorted_rows.length > 1) pitch = Math.max(1e-9, sorted_rows[1] - sorted_rows[0]);
      const lower = finite_rows.length ? Math.min(...finite_rows) - pitch / 2 : 0.5;
      const upper = finite_rows.length ? Math.max(...finite_rows) + pitch / 2 : 1.5;
      shapes.push({
        name: "thog2-centre-datum-background",
        type: "rect", xref: "x", yref: "y",
        x0: -0.5, x1: 0.5, y0: lower, y1: upper,
        line: {width: 0}, fillcolor: "#000000", layer: "above",
      });

      for (let row = 0; row < rows.length; row += 1) {
        const centre_loss = finite_number(current[row]);
        const cells = Array.isArray(customdata[row]) ? customdata[row] : [];
        let best = null;
        for (let column = 0; column < offsets.length; column += 1) {
          const delta = Array.isArray(cells[column]) ? finite_number(cells[column][3]) : null;
          if (delta === null || !(delta < 0) || centre_loss === null) continue;
          const loss = centre_loss + delta;
          if (best === null || loss < best.loss) best = {column, loss};
        }
        const decision_offset = Boolean(decisions[row])
          ? finite_number(Number(selected[row]) - Number(active[row]))
          : null;
        const decision_column = decision_offset === null
          ? -1
          : offsets.findIndex(offset => offset === decision_offset);
        if (decision_column >= 0) {
          shapes.push({
            name: "thog2-committed-decision-brick",
            type: "rect", xref: "x", yref: "y",
            x0: decision_offset - 0.5, x1: decision_offset + 0.5,
            y0: rows[row] - pitch / 2, y1: rows[row] + pitch / 2,
            line: {color: "#000000", width: 1}, fillcolor: "#ffffff", layer: "above",
          });
        }

        const labelled_columns = new Set();
        if (best) {
          annotations.push({
            name: "thog2-best-better-loss",
            x: offsets[best.column], y: rows[row], xref: "x", yref: "y",
            text: heatmap_cell_label(best.loss, centre_loss),
            showarrow: false, xanchor: "center", yanchor: "middle",
            font: {...font, color: "#000000"}, align: "center", captureevents: false,
          });
          labelled_columns.add(best.column);
        }
        if (decision_column >= 0 && !labelled_columns.has(decision_column)) {
          const delta = Array.isArray(cells[decision_column])
            ? finite_number(cells[decision_column][3])
            : null;
          const loss = centre_loss !== null && delta !== null ? centre_loss + delta : null;
          annotations.push({
            name: "thog2-committed-decision-text",
            x: offsets[decision_column], y: rows[row], xref: "x", yref: "y",
            text: heatmap_cell_label(loss, centre_loss),
            showarrow: false, xanchor: "center", yanchor: "middle",
            font: {...font, color: "#000000"}, align: "center", captureevents: false,
          });
        }
      }

      const latest_index = latest_meta_index(meta);
      if (Boolean(meta.thog2_brake_active?.[latest_index])) {
        annotations.push({
          name: "thog2-update-brake", x: 0.01, y: 1.205, xref: "paper", yref: "paper",
          text: "<b>update brake on</b>", showarrow: false,
          xanchor: "left", yanchor: "bottom", font: {size: 13, color: salmon},
        });
      }
      const chaos = meta.thog2_chaos_bump?.[latest_index];
      if (chaos?.state === "active") {
        annotations.push({
          name: "thog2-chaos-bump", x: 0.01, y: 1.145, xref: "paper", yref: "paper",
          text: (
            `sampling chaos bump made - magnitude ${Number(chaos.magnitude_percent).toFixed(1)}%. `
            + `Step ${chaos.step}/${chaos.duration}`
          ),
          showarrow: false, xanchor: "left", yanchor: "bottom",
          font: {size: 13, color: pale_blue},
        });
      } else if (chaos?.state === "reverted") {
        annotations.push({
          name: "thog2-chaos-bump", x: 0.01, y: 1.145, xref: "paper", yref: "paper",
          text: "reverted to pre-chaos bump sampling", showarrow: false,
          xanchor: "left", yanchor: "bottom", font: {size: 13, color: pale_blue},
        });
      }
      prepared.layout.annotations = annotations;
      prepared.layout.shapes = shapes;
    };

    const base_transpose_heatmap_v058 = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      base_transpose_heatmap_v058(prepared);
      const heatmap = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (!heatmap) return;
      const meta = prepared.layout?.meta || {};
      const latest_index = latest_meta_index(meta);
      const latest_l = finite_number(meta.thog2_active_layers?.[latest_index]);
      const offsets = Array.isArray(heatmap.x) ? heatmap.x.map(Number) : [];
      const previous_index = Math.max(0, latest_index - 1);
      const changed = latest_index > 0
        && Number(meta.thog2_selected_layers?.[previous_index]) !== Number(meta.thog2_active_layers?.[previous_index])
        && Number(meta.thog2_selected_layers?.[previous_index]) === latest_l;
      const relative_ticks = offsets.map(offset => (
        offset === 0 && latest_l !== null
          ? `<b style="color:${changed ? "#1769d2" : "#20252c"}">L=${latest_l}</b>`
          : signed_offset(offset)
      ));
      const absolute_ticks = offsets.map(offset => (
        latest_l === null ? "—" : String(latest_l + offset)
      ));
      const common_tick_font = {
        ...(prepared.layout?.xaxis?.tickfont || {}),
        size: Math.max(14, Number(prepared.layout?.xaxis?.tickfont?.size || 0)),
      };
      prepared.layout.xaxis = {
        ...(prepared.layout.xaxis || {}),
        side: "bottom", anchor: "y", tickmode: "array", tickvals: offsets,
        ticktext: relative_ticks,
        title: {text: "candidate layer-count offset from L", standoff: 18, font: {size: 14}},
        tickfont: common_tick_font,
      };
      prepared.layout.xaxis2 = {
        ...(prepared.layout.xaxis || {}),
        side: "top", anchor: "y", overlaying: "x", matches: "x",
        visible: true, showticklabels: true, showline: true, ticks: "outside",
        tickmode: "array", tickvals: offsets, ticktext: absolute_ticks,
        showgrid: false, zeroline: false,
        title: {
          text: latest_l === null
            ? "absolute candidate layer count"
            : `absolute candidate layer count · latest L=${latest_l}`,
          standoff: 16, font: {size: 14},
        },
        tickfont: common_tick_font,
      };
      prepared.layout.yaxis = {
        ...(prepared.layout.yaxis || {}),
        title: {text: "optimizer step", font: {size: 14}},
        tickfont: {
          ...(prepared.layout?.yaxis?.tickfont || {}),
          size: Math.max(14, Number(prepared.layout?.yaxis?.tickfont?.size || 0)),
        },
      };
      prepared.layout.margin = {
        ...(prepared.layout.margin || {}),
        t: Math.max(148, Number(prepared.layout?.margin?.t || 0)),
        b: Math.max(82, Number(prepared.layout?.margin?.b || 0)),
      };
      // The old active-layer line was extended to axis boundaries and could make
      // a blank protrusion outside the first heatmap row. Metadata now owns L.
      prepared.data = (prepared.data || []).filter(trace => trace === heatmap || trace.type === "heatmap");
      repair_heatmap_overlays(prepared, heatmap);
      top_axis_anchor(prepared, heatmap.x, heatmap.y);
      const colourbar = heatmap.colorbar || {};
      const title = typeof colourbar.title === "string"
        ? {text: colourbar.title}
        : {...(colourbar.title || {})};
      const title_text = String(title.text || "Δloss bands").replace(/(?:<br>)+$/g, "");
      heatmap.colorbar = {
        ...colourbar,
        title: {...title, text: `${title_text}<br>`, side: "top"},
      };
    };

    const collect_heatmap_settings = () => {
      if (app.axis_chart_name !== "heatmap" || !by_id("chart_heatmap_probe_count")) return null;
      const positive = (id, fallback) => {
        const value = finite_number(by_id(id)?.value);
        return value !== null && value > 0 ? value : fallback;
      };
      return {
        probe_count: Math.max(1, Math.min(512, Math.round(positive("chart_heatmap_probe_count", 100)))),
        window_mode: by_id("chart_heatmap_window_mode")?.value === "from_zero" ? "from_zero" : "rolling",
        y_display_mode: by_id("chart_heatmap_y_display_mode")?.value === "steps" ? "steps" : "probes",
        delta_loss_display_mode: by_id("chart_heatmap_delta_mode")?.value === "absolute" ? "absolute" : "percent",
        auto_colour_saturation: by_id("chart_heatmap_auto_colour")?.checked === true,
        abs_limit: positive("chart_heatmap_abs_limit", 0.05),
        negative_abs_limit: Math.min(0.1, positive("chart_heatmap_green_limit", 0.05)),
        blue_abs_limit: Math.min(1, Math.max(0.100000001, positive("chart_heatmap_blue_limit", 1))),
        yellow_abs_limit: Math.max(1.000000001, positive("chart_heatmap_yellow_limit", 2)),
        positive_abs_limit: positive("chart_heatmap_red_limit", 0.05),
      };
    };

    const save_button = by_id("save_chart_settings");
    save_button?.addEventListener("click", () => {
      pending_heatmap_settings = collect_heatmap_settings();
    }, true);
    save_button?.addEventListener("click", () => {
      const settings = pending_heatmap_settings;
      pending_heatmap_settings = null;
      if (!settings || !by_id("chart_settings_overlay")?.hidden) return;
      const previous = heatmap_settings_for_current_run();
      for (const [name, value] of Object.entries(settings)) {
        save_heatmap_viewer_setting(name, value);
      }
      const needs_refetch = (
        Number(previous.probe_count || 100) !== Number(settings.probe_count)
        || String(previous.window_mode || "rolling") !== settings.window_mode
      );
      if (needs_refetch) {
        const performance = window.__thog2_dashboard_performance?.state;
        if (performance) performance.heatmap_signature = null;
        app.figure_revision = null;
      }
      setTimeout(() => {
        const mount = by_id("heatmap_plot");
        if (needs_refetch) {
          const retry = () => {
            if (app.refresh_in_flight) {
              setTimeout(retry, 80);
              return;
            }
            refresh_current_run();
          };
          retry();
        }
        else if (mount && app.figures?.heatmap) render_plot(mount, app.figures.heatmap, "heatmap");
      }, 0);
    });

    const migrate_heatmap_title = () => {
      chart_titles.heatmap = heatmap_title;
      const stored = app.axis_ranges?.heatmap;
      if (stored && old_heatmap_titles.has(String(stored.title || ""))) {
        delete stored.title;
        if (!Object.keys(stored).length) delete app.axis_ranges.heatmap;
        save_json("thog2_local_chart_axis_ranges", app.axis_ranges);
      }
      document.querySelector('.chart-card[data-chart="heatmap"] .chart-heading-copy h2')
        ?.replaceChildren(heatmap_title);
    };
    migrate_heatmap_title();

    const base_render_figures_v058 = render_figures;
    render_figures = async function() {
      set_heatmap_group_visibility();
      const result = await base_render_figures_v058();
      install_weight_scale_controls();
      migrate_heatmap_title();
      if (app.workspace_mode) {
        const count = visible_runs().length;
        for (const chart_name of weight_chart_names) {
          const detail = by_id(`${chart_name}_detail`);
          if (detail && app.figures?.depth?.[chart_name]) {
            detail.textContent = `${count} visible run${count === 1 ? "" : "s"} · shared axes`;
          }
        }
        render_workspace_heading();
      }
      return result;
    };

    const weights_heading = by_id("coefficients_group_toggle")?.querySelector("strong");
    if (weights_heading) weights_heading.textContent = "weights";

    const style = document.createElement("style");
    style.textContent = `
      .workspace-icon { width: 22px; height: 18px; display: grid; grid-template-columns: repeat(3,1fr); gap: 2px; align-items: end; }
      .workspace-icon i { display: block; border-radius: 2px 2px 0 0; background: currentColor; }
      .workspace-icon i:nth-child(1) { height: 55%; }
      .workspace-icon i:nth-child(2) { height: 100%; }
      .workspace-icon i:nth-child(3) { height: 72%; }
      #heatmap_chart_group[aria-hidden="true"] { display: none !important; }
      .instra-workspace-mode #run_detail_tabs { visibility: hidden !important; }
      .instra-workspace-mode #heatmap_chart_group,
      .instra-workspace-mode .thog2-pending-train-group { display: none !important; }
      .instra-workspace-mode .selected-run-mark { border-radius: 4px; }
      .explicit-trajectory-modes { flex: 0 0 auto; }
      .chart-card[data-chart="heatmap"] .xtick text,
      .chart-card[data-chart="heatmap"] .ytick text { font-size: 14px !important; }
    `;
    document.head.appendChild(style);

    set_heatmap_group_visibility();
    if (app.figures?.heatmap && app.current_run_id) {
      queueMicrotask(async () => {
        const mount = by_id("heatmap_plot");
        if (mount) await render_plot(mount, app.figures.heatmap, "heatmap");
      });
    }
  }, 1250);
});
// ^^^ THOG
