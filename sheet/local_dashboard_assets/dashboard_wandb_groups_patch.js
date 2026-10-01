// vvv THOG
"use strict";

// W&B-like navigation for all scalar/history and native system charts found in
// the selected run's already-local .wandb database. Data are lazy-loaded only
// for expanded groups so a collapsed 20+ chart system group costs almost nothing.
window.addEventListener("load", () => {
  setTimeout(() => {
    // vvv THOG memory is a first-class group rather than a subset of device/system charts
    const group_order = new Map([["train", 0], ["val", 1], ["memory", 2], ["system", 3]]);
    // ^^^ THOG
    const group_revisions = new Map();
    const rendered_revisions = new Map();
    const collapsed_by_mode = new Map();
    let poll_view = null;
    let last_run_id = null;
    const front_by_chart = new Map();
    const pinned_bold_by_chart = new Map();                                                                                                                  // <<< THOG retain explicit !+ curve emphasis independently of z-order
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
      if (name === "train" && (app.instra_focus_loss === true || app.workspace_mode === true || workspace_api() || /(?:\?|&)runner_grid_tag=/.test(window.location?.search || ""))) return false;
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
      window.__instra_workspace?.cancel_pending?.();
      last_run_id=null;poll_view=null;
      const native = by_id("training_throughput_card"), legacy = by_id("training_grid");
      if (native && legacy && native.parentElement !== legacy) legacy.appendChild(native);
      for (const id of ["training_throughput_plot","processing_throughput_plot"]) {
        const mount = by_id(id);
        if (mount) clear_plot(mount);
      }
      delete app.dynamic_chart_figures.training_throughput;
      delete app.dynamic_chart_figures.processing_throughput;
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
        '<span class="group-caret" aria-hidden="true">⌄</span>'                                                                                          // <<< THOG remove the noninteractive six-dot motif
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
      const grid = section.querySelector(".local-metric-grid");
      section.querySelector(".local-metric-group-count").textContent = summary.name === "train"
        ? String(grid.querySelectorAll(".local-metric-card").length
            + Number(Boolean(grid.querySelector("#training_throughput_card"))))
        : String(summary.chart_count || 0);
      let message = section.querySelector(".local-metric-empty");
      if (!message) {
        message = document.createElement("p");
        message.className = "local-metric-empty";
        section.querySelector(".local-metric-grid").appendChild(message);
      }
      message.textContent = summary.reason || "No system metrics have been recorded for this run yet.";
      message.hidden = summary.name === "train" || Number(summary.chart_count || 0) > 0;
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
          if([...section.querySelectorAll(".local-metric-card")].some(card=>card.dataset.chart===app.maximized_chart))
            restore_maximized_chart();
          for(const card of section.querySelectorAll(".local-metric-card")) {
            const mount=card.querySelector(".plot-mount");
            if(mount?.dataset.plotReady==="true")Plotly.purge(mount);
            delete app.dynamic_chart_figures[card.dataset.chart];
            delete app.dynamic_chart_metadata[card.dataset.chart];delete chart_titles[card.dataset.chart];
          }
          group_revisions.delete(section.dataset.metricGroup);rendered_revisions.delete(section.dataset.metricGroup);
          section.remove();
        }
      }
      const ordered_sections = sorted_group_summaries(summaries).map(update_group_section);
      const train=group_section("train")?.querySelector(".local-metric-grid");
      if(train && !train.querySelector('[data-metric-chart-id="train/loss"]')) {
        const loss=make_metric_card("train",{id:"train/loss",title:"Loss",series:[]});
        loss.classList.add("instra-train-loss-card");
        loss.querySelector(".local-metric-detail").textContent="Loading loss for the selected runs…";
        train.prepend(loss);
      }
      // vvv THOG Processing is an ordinary chart group in the stack: after Val
      // (or Train when Val is unavailable), before Memory/System.  Reparent only
      // when this managed sequence actually changes; moving live Plotly nodes on
      // every one-second metadata poll caused visible flicker and stale widths.
      const processing_group = by_id("processing_chart_group");
      const processing_anchor = ordered_sections.find(section => section.dataset.metricGroup === "val")
        || ordered_sections.find(section => section.dataset.metricGroup === "train");
      const desired_nodes = [...ordered_sections];
      if (processing_group && processing_anchor) {
        desired_nodes.splice(desired_nodes.indexOf(processing_anchor) + 1, 0, processing_group);
      }
      const managed_nodes = new Set(desired_nodes);
      const current_nodes = [...parent.children].filter(node => managed_nodes.has(node));
      const order_changed = current_nodes.length !== desired_nodes.length
        || desired_nodes.some((node, index) => current_nodes[index] !== node);
      if (order_changed) {
        for (const node of desired_nodes) parent.insertBefore(node, depth_group || null);
      }
      // ^^^ THOG
    };

    // vvv THOG z-order controls make the front curve visually explicit and preserve independently pinned bold curves
    const ordered_metric_figure = (figure, chart_name) => {
      if (!workspace_api() || !figure?.data) return figure;
      const ids = [...new Set(figure.data.map(trace => trace.meta?.instra_workspace_run_id).filter(Boolean))];
      if (!ids.length) return figure;
      let front = front_by_chart.get(chart_name);
      if (!ids.includes(front)) {
        front = ids.at(-1);
        front_by_chart.set(chart_name, front);
      }
      const index = ids.indexOf(front);
      const order = [...ids.slice(index + 1), ...ids.slice(0, index + 1)];
      const rank = new Map(order.map((id, position) => [id, position]));
      const pinned = pinned_bold_by_chart.get(chart_name) || new Set();
      figure.data.sort((left, right) => (rank.get(left.meta?.instra_workspace_run_id) ?? -1) - (rank.get(right.meta?.instra_workspace_run_id) ?? -1));
      for (const trace of figure.data) {
        const run_id = trace.meta?.instra_workspace_run_id;
        trace.line = {...(trace.line || {}), width:(run_id === front || pinned.has(run_id)) ? 4.6 : 2.4};
      }
      window.instra_front_run_id = front;
      queueMicrotask(() => typeof render_runs === "function" && render_runs());
      return figure;
    };
    // ^^^ THOG
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
        // vvv THOG replace one-way z cycling with reverse cycling and explicit persistent bold/unbold controls
        const controls = document.createElement("div");
        controls.className = "metric-z-controls";
        const make_control = (label, title, action) => {
          const control = document.createElement("button");
          control.type = "button";
          control.className = "weight-step-button metric-z-cycle";
          control.textContent = label;
          control.title = title;
          control.setAttribute("aria-label", title);
          control.hidden = !workspace_api();
          control.addEventListener("click", async event => {
            event.stopPropagation();
            const figure = app.dynamic_chart_figures[key];
            const ids = [...new Set((figure?.data || []).map(trace => trace.meta?.instra_workspace_run_id).filter(Boolean))];
            if (!ids.length) return;
            let current = ids.includes(front_by_chart.get(key)) ? front_by_chart.get(key) : ids.at(-1);
            if (action === "next" || action === "previous") {
              const delta = action === "next" ? 1 : -1;
              current = ids[(ids.indexOf(current) + delta + ids.length) % ids.length];
              front_by_chart.set(key, current);
            } else {
              const pinned = pinned_bold_by_chart.get(key) || new Set();
              if (action === "pin") pinned.add(current);
              else pinned.delete(current);
              pinned_bold_by_chart.set(key, pinned);
            }
            window.instra_front_run_id = current;
            await render_plot(article.querySelector(".plot-mount"), figure, key);
          });
          controls.appendChild(control);
          return control;
        };
        make_control("!+", "Keep the current front curve bold when its z-order changes", "pin");
        make_control("!-", "Remove persistent bolding from the current front curve", "unpin");
        make_control("z+", "Bring the next Workspace run to the front", "next");
        make_control("z-", "Bring the previous Workspace run to the front", "previous");
        header.appendChild(controls);
        // ^^^ THOG
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
        type: "scatter",
        mode: series.x?.length === 1 ? "lines+markers" : "lines",
        meta: {instra_workspace_run_id: series.instra_workspace_run_id || app.current_run_id,
          instra_run_name: (app.runs || []).find(run=>run_identifier(run)===(series.instra_workspace_run_id || app.current_run_id))?.artifact_name ||
            (app.runs || []).find(run=>run_identifier(run)===(series.instra_workspace_run_id || app.current_run_id))?.run_name ||
            series.instra_workspace_run_id || app.current_run_id},
        x: Array.isArray(series.x) ? series.x : [],
        thog2_x_variants: series.x_variants || {},
        y: Array.isArray(series.y) ? series.y : [],
        name: series.name || chart.title || chart.id,
        customdata: series.point_sources || (series.y || []).map(() => "W&B"),
        hovertemplate: "<b>%{meta.instra_run_name}</b><br>step: %{x}<br>value: %{y:.6g}<extra></extra>",
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
      const requested_view = current_view_key();
      app.dynamic_chart_metadata[key] = {
        x_source: chart.x_title || "Step",
        x_label: chart.x_title || "step",
        y_source: chart.title || chart.id,
        y_label: chart.y_title || chart.title || chart.id,
        default_x_axis_mode: chart.default_x_axis_mode || null,
        available_x_axis_modes: chart.available_x_axis_modes || [],
      };
      const figure = metric_figure(article, chart);
      if (requested_view !== current_view_key()) return;
      const cycles = article.querySelectorAll(".metric-z-cycle");
      if (cycles.length) {
        const count = new Set(figure.data.map(trace => trace.meta?.instra_workspace_run_id).filter(Boolean)).size;
        for (const cycle of cycles) {
          cycle.hidden = !workspace_api();
          cycle.disabled = cycle.textContent.startsWith("z") ? count < 2 : count < 1;
        }
      }
      app.dynamic_chart_figures[key] = figure;
      await render_plot(mount, figure, key);
      if (requested_view !== current_view_key()) return;
      const detail = article.querySelector(".local-metric-detail");
      if (detail) {
        const count = point_count(chart);
        const series_count = figure.data.length;
        detail.textContent = `${format_integer(count)} sample${count === 1 ? "" : "s"}${series_count > 1 ? ` · ${series_count} series` : ""}`;
      }
    };

    const render_group_payload = async (payload, requested_view) => {
      if (requested_view !== current_view_key()) return;
      const group = payload?.group;
      if (!group || !group.name) return;
      const section = group_section(group.name);
      const grid = section?.querySelector(".local-metric-grid");
      if (!section || !grid) return;
      // The native Train throughput plot already lives in this grid. The
      // recorded throughput history must not displace the loss plot.
      const charts = (group.charts || []).filter(chart => (chart.series || []).some(series=>series.x?.length && series.y?.length))
        .filter(chart => !(group.name === "train" &&
        document.getElementById("training_throughput_card") &&
        /(?:token.*(?:sec|throughput)|throughput)/i.test(`${chart.id} ${chart.title}`)))
        .sort((left,right)=>group.name==="train" ? Number(right.id==="train/loss")-Number(left.id==="train/loss") : 0);
      const wanted = new Set(charts.map(chart => chart.id));
      for (const card of [...grid.querySelectorAll(".local-metric-card")]) {
        if (!wanted.has(card.dataset.metricChartId)) {
          if(group.name==="train" && card.dataset.metricChartId==="train/loss")continue;
          if(card.dataset.chart===app.maximized_chart)restore_maximized_chart();
          const mount = card.querySelector(".plot-mount");
          if (mount?.dataset.plotReady === "true") Plotly.purge(mount);
          const key = card.dataset.chart;
          delete app.dynamic_chart_figures[key];
          delete app.dynamic_chart_metadata[key];
          delete chart_titles[key];
          card.remove();
        }
      }

      const render_jobs = [];
      for (const chart of charts) {
        let card = [...grid.querySelectorAll(".local-metric-card")].find(candidate => candidate.dataset.metricChartId === chart.id);
        if (!card) {
          card = make_metric_card(group.name, chart);
          grid.appendChild(card);
        }
        if (chart.id === "train/loss" && grid.firstElementChild !== card) grid.prepend(card);
        render_jobs.push(() => requested_view===current_view_key() ? render_metric_chart(card, chart) : Promise.resolve());
      }
      // Drawing a dozen Memory or System plots serially adds all individual
      // Plotly startup times. Keep a small bound to preserve UI responsiveness.
      for (let start = 0; start < render_jobs.length; start += 3) {
        await Promise.all(render_jobs.slice(start, start + 3).map(render => render()));
        if (requested_view !== current_view_key()) return;
      }
      if (group.name === "train") {
        section.querySelector(".local-metric-group-count").textContent = String(
          grid.querySelectorAll(".local-metric-card").length
          + Number(Boolean(grid.querySelector("#training_throughput_card")))
        );
      }
      if (group.name === "train" && (app.workspace_mode === true || app.instra_focus_loss === true) && !app.instra_loss_autofocused &&
          !app.maximized_chart && charts.some(chart => chart.id === "train/loss")) {
        const loss = [...grid.querySelectorAll(".local-metric-card")].find(card => card.dataset.metricChartId === "train/loss");
        if (loss) {
          app.instra_loss_autofocused = true;
          app.instra_focus_loss = false;
          toggle_maximized_chart(loss.dataset.chart);
        }
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
          ? await workspace.fetch_metric_group(group_name,group_name==="train" ? partial=>render_group_payload(partial,requested_view) : null)
          : await fetch_json(
              `/api/chart-group?run=${encodeURIComponent(app.current_run_id)}`
              + `&group=${encodeURIComponent(group_name)}`
            );
        if (requested_view !== current_view_key()) return;
        if (payload.available === false) return;
        await render_group_payload(payload, requested_view);
      } catch (error) {
        if(requested_view===current_view_key() && error.name!=="AbortError")show_toast(`Chart group ${group_name} failed: ${error.message}`);
      }
    }

    const refresh_metric_groups = async () => {
      const requested_run = current_view_key();
      if (!app.current_run_id || poll_view?.view===requested_run) return;
      // vvv THOG pause chart discovery in a hidden tab; catch up as soon as it is visible
      if (document.visibilityState === "hidden" ||
          typeof instra_charts_visible === "function" && !instra_charts_visible()) return;
      // ^^^ THOG
      if (by_id("charts_scroll")?.hidden) return;
      const owner={view:requested_run};poll_view=owner;
      try {
        const changing_view=last_run_id !== requested_run;
        if (changing_view) {
          invalidate_metric_groups(true);
          last_run_id = requested_run;
        }
        const workspace = workspace_api();
        let early_train=null;
        if(workspace && changing_view) {
          sync_group_order([{name:"train",chart_count:0,revision:0},{name:"val",chart_count:0,revision:0},{name:"system",chart_count:0,revision:0}]);
          early_train=refresh_group_data("train",true);
        }
        const payload = workspace
          ? await workspace.fetch_metric_groups()
          : await fetch_json(`/api/chart-groups?run=${encodeURIComponent(app.current_run_id)}`);
        if (requested_run !== current_view_key()) return;
        const summaries = (payload.groups || []).filter(summary => summary.name !== "depth" &&
          (summary.name!=="plastic" || Number(summary.chart_count)>0));
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
        const opened = summaries.filter(summary => {
          const section = group_section(summary.name);
          return section && !section.classList.contains("collapsed");
        });
        // vvv THOG let Train render while large Memory/System groups load independently
        await Promise.all([...opened.filter(summary=>!early_train || summary.name!=="train").map(summary => refresh_group_data(summary.name)),early_train]);
        // ^^^ THOG
        restore_metric_navigation();
      } catch (error) {
        if(requested_run===current_view_key() && error.name!=="AbortError")show_toast(`Local W&B charts failed: ${error.message}`);
      } finally {
        if(poll_view===owner)poll_view=null;
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
      .metric-z-controls { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); display:flex; gap:4px; z-index:5; }
      .metric-z-cycle { position:static; transform:none; margin:0; min-width:28px; }
      .local-metric-card.maximized > .chart-card-header { position: relative !important; display: flex !important; visibility: visible !important; }
      .local-metric-card.maximized > .chart-card-header > .metric-z-controls { display:flex !important; visibility:visible !important; opacity:1 !important; }
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
    // vvv THOG resume chart discovery promptly after background tab quiescence
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") refresh_metric_groups();
    });
    // ^^^ THOG
    setTimeout(refresh_metric_groups, 50);
  }, 0);
});
// ^^^ THOG
