// vvv THOG
"use strict";

// W&B-like navigation for all scalar/history and native system charts found in
// the selected run's already-local .wandb database. Data are lazy-loaded only
// for expanded groups so a collapsed 20+ chart system group costs almost nothing.
window.addEventListener("load", () => {
  setTimeout(() => {
    const group_order = new Map([["train", 0], ["val", 1], ["system", 2]]);
    const group_revisions = new Map();
    const rendered_revisions = new Map();
    let collapsed_group_context = "";
    let collapsed_group_settings = new Map();
    let poll_in_flight = false;
    let last_run_id = null;

    const workspace_api = () => {
      const candidate = window.__instra_workspace;
      return candidate?.active?.() === true ? candidate : null;
    };
    const current_view_key = () => workspace_api()?.selection_key?.() || app.current_run_id;
    const group_context_key = () => {
      const workspace = workspace_api();
      if (workspace) {
        const run_ids = (workspace.visible_runs?.() || [])
          .map(run => String(run_identifier(run)))
          .filter(Boolean)
          .sort();
        return `workspace:${run_ids.join("|")}`;
      }
      return `run:${String(app.current_run_id || "")}`;
    };

    const metric_group_sections = () => [...document.querySelectorAll(".local-metric-group")];
    const group_section = name => metric_group_sections().find(section => section.dataset.metricGroup === name) || null;
    const group_collapsed_settings = (key = group_context_key()) => {
      if (key !== collapsed_group_context) {
        collapsed_group_context = key;
        collapsed_group_settings = new Map();
      }
      return collapsed_group_settings;
    };

    const group_is_collapsed = name => {
      const settings = group_collapsed_settings();
      return settings.has(name) ? settings.get(name) : true;
    };
    const save_group_collapsed = (name, collapsed) => {
      const settings = group_collapsed_settings();
      settings.set(name, Boolean(collapsed));
    };

    const chart_key = (group_name, chart_id) => `local_metric_${hash_text(`${group_name}\0${chart_id}`).toString(16)}`;
    const group_key = group_name => `local_metric_group_${hash_text(group_name).toString(16)}`;

    const clear_metric_groups = () => {
      if (app.maximized_chart && String(app.maximized_chart).startsWith("local_metric_")) restore_maximized_chart();
      for (const section of metric_group_sections()) {
        for (const mount of section.querySelectorAll(".plot-mount")) {
          if (mount.dataset.plotReady === "true") Plotly.purge(mount);
        }
        for (const card of section.querySelectorAll(".local-metric-card")) {
          const key = card.dataset.chart;
          delete app.dynamic_chart_figures[key];
          delete app.dynamic_chart_metadata[key];
          delete chart_titles[key];
        }
        section.remove();
      }
      group_revisions.clear();
      rendered_revisions.clear();
    };

    const sorted_group_summaries = groups => [...groups].sort((left, right) => {
      const left_order = group_order.has(left.name) ? group_order.get(left.name) : 100;
      const right_order = group_order.has(right.name) ? group_order.get(right.name) : 100;
      if (left_order !== right_order) return left_order - right_order;
      return String(left.name).localeCompare(String(right.name));
    });

    const make_group_section = summary => {
      const section = document.createElement("section");
      section.className = "chart-group local-metric-group";
      section.id = group_key(summary.name);
      section.dataset.chartGroup = summary.name;
      section.dataset.metricGroup = summary.name;

      const header = document.createElement("header");
      header.className = "chart-group-header";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "chart-group-toggle";
      button.setAttribute("aria-controls", `${section.id}_grid`);
      button.innerHTML = (
        '<span class="group-grip" aria-hidden="true">⠿</span>'
        + '<span class="group-caret" aria-hidden="true">⌄</span>'
      );
      const name = document.createElement("strong");
      name.textContent = summary.name;
      const count = document.createElement("span");
      count.className = "group-count local-metric-group-count";
      count.textContent = String(summary.chart_count || 0);
      button.append(name, count);
      header.appendChild(button);

      const grid = document.createElement("div");
      grid.className = "chart-grid local-metric-grid";
      grid.id = `${section.id}_grid`;

      const collapsed = group_is_collapsed(summary.name);
      section.classList.toggle("collapsed", collapsed);
      grid.hidden = collapsed;
      button.setAttribute("aria-expanded", String(!collapsed));
      button.addEventListener("click", () => {
        // The established dashboard click handler performs the actual toggle.
        queueMicrotask(() => {
          const now_collapsed = section.classList.contains("collapsed");
          save_group_collapsed(summary.name, now_collapsed);
          if (!now_collapsed) refresh_group_data(summary.name, true);
        });
      });

      section.append(header, grid);
      return section;
    };

    const update_group_section = summary => {
      let section = group_section(summary.name);
      if (!section) section = make_group_section(summary);
      section.querySelector(".local-metric-group-count").textContent = String(summary.chart_count || 0);
      group_revisions.set(summary.name, Number(summary.revision || 0));
      return section;
    };

    const sync_group_order = summaries => {
      const depth_group = by_id("depth_chart_group");
      const parent = depth_group?.parentElement || by_id("charts_scroll");
      if (!parent) return;
      const wanted = new Set(summaries.map(summary => summary.name));
      for (const section of metric_group_sections()) {
        if (!wanted.has(section.dataset.metricGroup)) section.remove();
      }
      for (const summary of sorted_group_summaries(summaries)) {
        const section = update_group_section(summary);
        parent.insertBefore(section, depth_group || null);
      }
    };

    const make_metric_card = (group_name, chart) => {
      const key = chart_key(group_name, chart.id);
      chart_titles[key] = chart.title || chart.id;
      app.dynamic_chart_metadata[key] = {
        x_source: chart.x_title || "Step",
        x_label: chart.x_title || "step",
        y_source: chart.title || chart.id,
        y_label: chart.y_title || chart.title || chart.id,
        default_x_axis_mode: chart.default_x_axis_mode || null,
        available_x_axis_modes: chart.available_x_axis_modes || [],
      };

      const article = document.createElement("article");
      article.className = "chart-card local-metric-card";
      article.dataset.chart = key;
      article.dataset.metricChartId = chart.id;
      article.dataset.metricGroup = group_name;

      const header = document.createElement("header");
      header.className = "chart-card-header";
      const copy = document.createElement("div");
      copy.className = "chart-heading-copy";
      const title = document.createElement("h2");
      title.textContent = normalize_chart_settings(key).title;
      const detail = document.createElement("p");
      detail.className = "local-metric-detail";
      copy.append(title, detail);
      const maximize = document.createElement("button");
      maximize.type = "button";
      maximize.className = "maximize-button";
      maximize.dataset.maximize = key;
      maximize.textContent = "⛶";
      maximize.title = "Maximize chart";
      maximize.setAttribute("aria-label", `Maximize ${title.textContent}`);
      const actions = document.createElement("div");
      actions.className = "chart-card-actions";
      actions.append(chart_settings_button(key, title.textContent), maximize);
      header.append(copy, actions);

      const shell = document.createElement("div");
      shell.className = "plot-shell";
      const mount = document.createElement("div");
      mount.className = "plot-mount";
      mount.id = `${key}_plot`;
      shell.appendChild(mount);
      article.append(header, shell);
      add_panel_resizers(article);
      return article;
    };

    const point_count = chart => Math.max(0, ...(chart.series || []).map(series => Number(series.points || series.x?.length || 0)));

    const metric_figure = (article, chart) => {
      const traces = (chart.series || []).map((series, index) => ({
        type: "scattergl",
        mode: "lines",
        x: Array.isArray(series.x) ? series.x : [],
        thog2_x_variants: series.x_variants || {},
        y: Array.isArray(series.y) ? series.y : [],
        name: series.name || chart.title || chart.id,
        hovertemplate: "%{y:.6g}<extra>%{fullData.name}</extra>",
        line: {
          width: 1.35,
          color: series.color || default_palette[index % default_palette.length],
        },
      }));
      const multi_series = traces.length > 1;
      const layout = {
        autosize: true,
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        hovermode: multi_series ? "x unified" : "closest",
        showlegend: multi_series,
        margin: {l: 58, r: 18, t: 16, b: 48},
        xaxis: {
          title: {text: chart.x_title || "step", standoff: 8},
          automargin: true,
          gridcolor: "#e7ebf0",
          zerolinecolor: "#c8ced6",
        },
        yaxis: {
          automargin: true,
          gridcolor: "#e7ebf0",
          zerolinecolor: "#c8ced6",
        },
        legend: {
          x: 1,
          xanchor: "right",
          y: 1,
          yanchor: "top",
          bgcolor: "rgba(255,255,255,.72)",
          font: {size: 9},
        },
        uirevision: `${app.current_run_id}-${article.dataset.chart}`,
        font: {family: "Inter, ui-sans-serif, system-ui, sans-serif", size: 10, color: "#35404c"},
      };
      return {data: traces, layout};
    };

    const render_metric_chart = async (article, chart) => {
      const mount = article.querySelector(".plot-mount");
      if (!mount) return;
      const key = article.dataset.chart;
      app.dynamic_chart_metadata[key] = {
        x_source: chart.x_title || "Step",
        x_label: chart.x_title || "step",
        y_source: chart.title || chart.id,
        y_label: chart.y_title || chart.title || chart.id,
        default_x_axis_mode: chart.default_x_axis_mode || null,
        available_x_axis_modes: chart.available_x_axis_modes || [],
      };
      const figure = metric_figure(article, chart);
      app.dynamic_chart_figures[key] = figure;
      await render_plot(mount, figure, key);
      const detail = article.querySelector(".local-metric-detail");
      if (detail) {
        const count = point_count(chart);
        const series_count = figure.data.length;
        detail.textContent = `${format_integer(count)} sample${count === 1 ? "" : "s"}${series_count > 1 ? ` · ${series_count} series` : ""}`;
      }
    };

    const render_group_payload = async payload => {
      const group = payload?.group;
      if (!group || !group.name) return;
      const section = group_section(group.name);
      const grid = section?.querySelector(".local-metric-grid");
      if (!section || !grid) return;

      const wanted = new Set((group.charts || []).map(chart => chart.id));
      for (const card of [...grid.querySelectorAll(".local-metric-card")]) {
        if (!wanted.has(card.dataset.metricChartId)) {
          const mount = card.querySelector(".plot-mount");
          if (mount?.dataset.plotReady === "true") Plotly.purge(mount);
          const key = card.dataset.chart;
          delete app.dynamic_chart_figures[key];
          delete app.dynamic_chart_metadata[key];
          delete chart_titles[key];
          card.remove();
        }
      }

      for (const chart of group.charts || []) {
        let card = [...grid.querySelectorAll(".local-metric-card")].find(candidate => candidate.dataset.metricChartId === chart.id);
        if (!card) {
          card = make_metric_card(group.name, chart);
          grid.appendChild(card);
        }
        await render_metric_chart(card, chart);
      }
      rendered_revisions.set(group.name, Number(group.revision || 0));
      apply_saved_panel_sizes();
      requestAnimationFrame(resize_visible_plots);
    };

    async function refresh_group_data(group_name, force = false) {
      if (!app.current_run_id) return;
      const section = group_section(group_name);
      if (!section || section.classList.contains("collapsed")) return;
      const revision = Number(group_revisions.get(group_name) || 0);
      if (!force && rendered_revisions.get(group_name) === revision) return;
      const requested_view = current_view_key();
      try {
        const workspace = workspace_api();
        const payload = workspace
          ? await workspace.fetch_metric_group(group_name)
          : await fetch_json(
              `/api/chart-group?run=${encodeURIComponent(app.current_run_id)}`
              + `&group=${encodeURIComponent(group_name)}`
            );
        if (requested_view !== current_view_key()) return;
        if (payload.available === false) return;
        await render_group_payload(payload);
      } catch (error) {
        show_toast(`Chart group ${group_name} failed: ${error.message}`);
      }
    }

    const refresh_metric_groups = async () => {
      if (!app.current_run_id || poll_in_flight) return;
      if (by_id("charts_scroll")?.hidden) return;
      poll_in_flight = true;
      const requested_run = current_view_key();
      try {
        if (last_run_id !== requested_run) {
          clear_metric_groups();
          last_run_id = requested_run;
        }
        const workspace = workspace_api();
        const payload = workspace
          ? await workspace.fetch_metric_groups()
          : await fetch_json(`/api/chart-groups?run=${encodeURIComponent(app.current_run_id)}`);
        if (requested_run !== current_view_key()) return;
        if (!payload.available) {
          clear_metric_groups();
          return;
        }
        const summaries = (payload.groups || []).filter(summary => summary.name !== "depth");
        sync_group_order(summaries);
        for (const summary of summaries) {
          const section = group_section(summary.name);
          if (section && !section.classList.contains("collapsed")) {
            await refresh_group_data(summary.name);
          }
        }
      } catch (error) {
        show_toast(`Local W&B charts failed: ${error.message}`);
      } finally {
        poll_in_flight = false;
      }
    };

    const base_select_run_metric_groups = select_run;
    select_run = function(run_id, options = {}) {
      if (run_id !== app.current_run_id) {
        clear_metric_groups();
        last_run_id = run_id;
      }
      const result = base_select_run_metric_groups(run_id, options);
      setTimeout(refresh_metric_groups, 0);
      return result;
    };

    const base_local_apply_detail_tab_metric_groups = typeof local_apply_detail_tab === "function"
      ? local_apply_detail_tab
      : null;
    if (base_local_apply_detail_tab_metric_groups) {
      local_apply_detail_tab = function() {
        const result = base_local_apply_detail_tab_metric_groups();
        if (!by_id("charts_scroll")?.hidden) setTimeout(refresh_metric_groups, 0);
        return result;
      };
    }

    const style = document.createElement("style");
    style.textContent = `
      .local-metric-group { min-height: 35px; }
      .local-metric-group:not(.collapsed) { min-height: 0; }
      .local-metric-group .chart-group-header { position: sticky; top: 0; z-index: 5; }
      .local-metric-group .local-metric-grid { min-height: 0; padding-top: 10px; }
      .local-metric-card { flex: 1 1 calc(33.333% - 10px); min-width: 300px; height: 365px; }
      .local-metric-card .plot-mount { width: 100%; min-width: 0; height: 100%; min-height: 0; }
      .local-metric-card .plot-shell { overflow: hidden; }
      .local-metric-card .modebar { opacity: .72; }
      .local-metric-card:hover .modebar { opacity: 1; }
      @media (max-width: 1100px) {
        .local-metric-card { flex-basis: calc(50% - 10px); }
      }
      @media (max-width: 760px) {
        .local-metric-card { flex-basis: 100%; }
      }
    `;
    document.head.appendChild(style);

    window.__thog2_metric_groups = {
      clear: clear_metric_groups,
      refresh: refresh_metric_groups,
      refresh_group: refresh_group_data,
      context_key: group_context_key,
      group_is_collapsed,
      set_group_collapsed: save_group_collapsed,
    };

    setInterval(refresh_metric_groups, 2500);
    setTimeout(refresh_metric_groups, 50);
  }, 0);
});
// ^^^ THOG
