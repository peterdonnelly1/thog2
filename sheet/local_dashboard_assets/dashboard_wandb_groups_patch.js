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
    const collapsed_by_mode = new Map();
    let poll_in_flight = false;
    let last_run_id = null;
    const front_by_chart = new Map();
    let pending_navigation = null;

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
    const group_collapsed_settings = () => {
      // Rebuilt groups inherit the user's choices across runs / Workspace membership changes.
      const key = workspace_api() ? "workspace" : "runs";
      if (!collapsed_by_mode.has(key)) {
        const stored = load_json("thog2_local_metric_group_collapsed_v2", {});
        collapsed_by_mode.set(key, new Map(Object.entries(stored[key] || {})));
      }
      return collapsed_by_mode.get(key);
    };

    const group_is_collapsed = name => {
      const settings = group_collapsed_settings();
      return settings.has(name) ? settings.get(name) : true;
    };
    const save_group_collapsed = (name, collapsed) => {
      const settings = group_collapsed_settings();
      settings.set(name, Boolean(collapsed));
      const stored = load_json("thog2_local_metric_group_collapsed_v2", {});
      stored[workspace_api() ? "workspace" : "runs"] = Object.fromEntries(settings);
      save_json("thog2_local_metric_group_collapsed_v2", stored);
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

    const invalidate_metric_groups = (clear_values = false) => {
      rendered_revisions.clear();
      if (!clear_values) return;
      group_revisions.clear();
      for (const section of metric_group_sections()) {
        for (const card of section.querySelectorAll(".local-metric-card")) {
          delete app.dynamic_chart_figures[card.dataset.chart];
          const mount = card.querySelector(".plot-mount");
          if (mount) clear_plot(mount);
          const detail = card.querySelector(".local-metric-detail");
          if (detail) detail.textContent = "Waiting for this run's data…";
        }
      }
    };

    const remember_metric_navigation = () => {
      const viewport = by_id("charts_scroll");
      const viewport_top = viewport?.getBoundingClientRect?.().top ?? 0;
      const sections = [...document.querySelectorAll(".chart-group")];
      const anchor = sections.find(section => section.getBoundingClientRect?.().bottom > viewport_top + 5);
      for (const section of metric_group_sections()) {
        save_group_collapsed(section.dataset.metricGroup, section.classList.contains("collapsed"));
      }
      return {
        chart_name: String(app.maximized_chart || "").startsWith("local_metric_") ? app.maximized_chart : pending_navigation?.chart_name || null,
        group_name: anchor?.dataset.chartGroup || pending_navigation?.group_name || null,
        offset: anchor ? anchor.getBoundingClientRect().top - viewport_top : pending_navigation?.offset || 0,
        scroll_top: Number(viewport?.scrollTop || 0),
      };
    };
    const restore_metric_navigation = () => {
      const saved = pending_navigation;
      if (!saved || saved.view !== current_view_key()) return;
      requestAnimationFrame(() => {
        if (saved !== pending_navigation || saved.view !== current_view_key()) return;
        const viewport = by_id("charts_scroll");
        if (!viewport) return;
        if (saved.chart_name) {
          const card = document.querySelector(`.chart-card[data-chart="${CSS.escape(saved.chart_name)}"]`);
          if (!card) return;
          if (!app.maximized_chart) toggle_maximized_chart(saved.chart_name);
        } else if (!app.maximized_chart) {
          const anchor = [...document.querySelectorAll(".chart-group")].find(section => section.dataset.chartGroup === saved.group_name);
          const viewport_top = viewport.getBoundingClientRect?.().top ?? 0;
          viewport.scrollTop = anchor
            ? viewport.scrollTop + anchor.getBoundingClientRect().top - viewport_top - saved.offset
            : saved.scroll_top;
        }
        pending_navigation = null;
      });
    };
    // A deliberate chart interaction supersedes a delayed navigation restoration.
    by_id("charts_scroll")?.addEventListener?.("pointerdown", () => { pending_navigation = null; }, true);
    by_id("charts_scroll")?.addEventListener?.("wheel", () => { pending_navigation = null; }, {passive: true});

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
      let message = section.querySelector(".local-metric-empty");
      if (!message) {
        message = document.createElement("p");
        message.className = "local-metric-empty";
        section.querySelector(".local-metric-grid").appendChild(message);
      }
      message.textContent = summary.reason || "No system metrics have been recorded for this run yet.";
      message.hidden = Number(summary.chart_count || 0) > 0;
      group_revisions.set(summary.name, Number(summary.revision || 0));
      return section;
    };

    const sync_group_order = summaries => {
      const depth_group = by_id("depth_chart_group");
      const parent = depth_group?.parentElement || by_id("charts_scroll");
      if (!parent) return;
      const wanted = new Set(summaries.map(summary => summary.name));
      for (const section of metric_group_sections()) {
        if (!wanted.has(section.dataset.metricGroup)) {
          update_group_section({name: section.dataset.metricGroup, chart_count: 0,
            reason: "No data recorded for this run yet."});
        }
      }
      for (const summary of sorted_group_summaries(summaries)) {
        const section = update_group_section(summary);
        parent.insertBefore(section, depth_group || null);
      }
    };

    const ordered_metric_figure = (figure, chart_name) => {
      if (!workspace_api() || !figure?.data) return figure;
      const ids = [...new Set(figure.data.map(trace => trace.meta?.instra_workspace_run_id).filter(Boolean))];
      const front = front_by_chart.get(chart_name);
      if (!ids.includes(front)) return figure;
      const index = ids.indexOf(front);
      const order = [...ids.slice(index + 1), ...ids.slice(0, index + 1)];
      const rank = new Map(order.map((id, position) => [id, position]));
      figure.data.sort((left, right) => (rank.get(left.meta?.instra_workspace_run_id) ?? -1) - (rank.get(right.meta?.instra_workspace_run_id) ?? -1));
      return figure;
    };
    const base_prepare_metric_order = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_metric_order(figure, chart_name);
      return String(chart_name).startsWith("local_metric_") ? ordered_metric_figure(prepared, chart_name) : prepared;
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
      maximize.innerHTML = chart_size_icon();
      maximize.title = "Maximize chart";
      maximize.setAttribute("aria-label", `Maximize ${title.textContent}`);
      const actions = document.createElement("div");
      actions.className = "chart-card-actions";
      actions.append(chart_settings_button(key, title.textContent), maximize);
      header.append(copy, actions);
      if (["train", "val"].includes(group_name)) {
        const cycle = document.createElement("button");
        cycle.type = "button";
        cycle.className = "weight-step-button metric-z-cycle";
        cycle.textContent = "z";
        cycle.title = "Bring the next Workspace run to the front";
        cycle.setAttribute("aria-label", cycle.title);
        cycle.hidden = !workspace_api();
        cycle.addEventListener("click", async event => {
          event.stopPropagation();
          const figure = app.dynamic_chart_figures[key];
          const ids = [...new Set((figure?.data || []).map(trace => trace.meta?.instra_workspace_run_id).filter(Boolean))];
          if (ids.length < 2) return;
          const current = ids.includes(front_by_chart.get(key)) ? front_by_chart.get(key) : ids.at(-1);
          front_by_chart.set(key, ids[(ids.indexOf(current) + 1) % ids.length]);
          await render_plot(article.querySelector(".plot-mount"), figure, key);
        });
        header.appendChild(cycle);
      }

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
        mode: series.x?.length === 1 ? "lines+markers" : "lines",
        meta: {instra_workspace_run_id: series.instra_workspace_run_id || app.current_run_id},
        x: Array.isArray(series.x) ? series.x : [],
        thog2_x_variants: series.x_variants || {},
        y: Array.isArray(series.y) ? series.y : [],
        name: series.name || chart.title || chart.id,
        customdata: series.point_sources || (series.y || []).map(() => "W&B"),
        hovertemplate: "%{x}<br>%{y:.6g}<br>%{customdata}<extra>%{fullData.name}</extra>",
        line: {
          width: 2.4,
          color: series.color || default_palette[index % default_palette.length],
        },
      }));
      const multi_series = traces.length > 1;
      const layout = {
        autosize: true,
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        hovermode: "closest",
        spikedistance: -1,
        showlegend: multi_series,
        margin: {l: 58, r: 18, t: 16, b: 48},
        xaxis: {
          title: {text: chart.x_title || "step", standoff: 8},
          showspikes: true, spikemode: "across", spikesnap: "cursor",
          spikecolor: "#555", spikethickness: 1, spikedash: "dot",
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
      const cycle = article.querySelector(".metric-z-cycle");
      if (cycle) {
        cycle.hidden = !workspace_api();
        cycle.disabled = new Set(figure.data.map(trace => trace.meta?.instra_workspace_run_id).filter(Boolean)).size < 2;
      }
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
          invalidate_metric_groups(true);
          last_run_id = requested_run;
        }
        const workspace = workspace_api();
        const payload = workspace
          ? await workspace.fetch_metric_groups()
          : await fetch_json(`/api/chart-groups?run=${encodeURIComponent(app.current_run_id)}`);
        if (requested_run !== current_view_key()) return;
        const summaries = (payload.groups || []).filter(summary => summary.name !== "depth");
        for (const name of ["train", "val"]) {
          if (!summaries.some(summary => summary.name === name)) summaries.push({
            name, chart_count: 0, revision: 0, reason: `Waiting for ${name} data…`,
          });
        }
        if (!summaries.some(summary => summary.name === "system")) {
          const reason = payload.error ? `Cannot read system metrics: ${payload.error}`
            : payload.reason ? `System metrics unavailable: ${payload.reason}.`
            : payload.catching_up ? "Loading system metrics from the local run file…"
            : "No system metrics recorded yet. W&B system monitoring must be enabled and its local run file accessible.";
          summaries.push({name: "system", chart_count: 0, revision: 0, reason});
        }
        sync_group_order(summaries);
        for (const summary of summaries) {
          const section = group_section(summary.name);
          if (section && !section.classList.contains("collapsed")) {
            await refresh_group_data(summary.name);
          }
        }
        restore_metric_navigation();
      } catch (error) {
        show_toast(`Local W&B charts failed: ${error.message}`);
      } finally {
        poll_in_flight = false;
      }
    };

    const base_select_run_metric_groups = select_run;
    select_run = function(run_id, options = {}) {
      const saved = run_id !== app.current_run_id ? remember_metric_navigation() : null;
      if (saved) {
        invalidate_metric_groups(true);
        last_run_id = null;
      }
      const result = base_select_run_metric_groups(run_id, options);
      if (saved) pending_navigation = {...saved, view: current_view_key()};
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
      .local-metric-card > .chart-card-header { position: relative; }
      .local-metric-card > .chart-card-header > .chart-heading-copy { max-width: calc(50% - 24px); }
      .metric-z-cycle { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); margin: 0; z-index: 5; }
      .local-metric-card.maximized > .chart-card-header { position: relative !important; display: flex !important; visibility: visible !important; }
      .local-metric-card.maximized > .chart-card-header > .metric-z-cycle:not([hidden]) { display: inline-flex !important; visibility: visible !important; opacity: 1 !important; }
      .metric-z-cycle[hidden] { display: none !important; }
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
      invalidate: invalidate_metric_groups,
      refresh: refresh_metric_groups,
      refresh_group: refresh_group_data,
      context_key: group_context_key,
      group_is_collapsed,
      set_group_collapsed: save_group_collapsed,
    };

    // A newly-opened active run may acquire its first committed W&B history record
    // between ordinary dashboard polls. One-second discovery keeps the pending
    // train group responsive without loading any collapsed chart payloads.
    setInterval(refresh_metric_groups, 1000);
    setTimeout(refresh_metric_groups, 50);
  }, 0);
});
// ^^^ THOG
