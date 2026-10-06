// vvv THOG bounded width capture belongs to the selected run or ordinary Multiview cohort
"use strict";
(function install_width_charts() {
  const chart_names = ["width_energy", "width_probes", "width_curves"];
  const titles = ["Residual mode energy", "Retained-prefix validation loss probes", "Sampled represented feature curves"];
  const axis_labels = [["Retained basis mode", "Mean squared coefficient"],
    ["Retained prefix count", "Delta validation cross-entropy (positive = harm)"],
    ["Reference feature index (bounded sample)", "Represented feature value"]];
  let owner = null, rendered_signature = null;
  const selected_steps = new Map(), selected_sites = new Map();
  const run_name = run => String(run.artifact_name || run.run_name || run_identifier(run));
  function visible_runs() {
    const runs = app.workspace_mode ? window.__instra_workspace?.visible_runs?.() || [] : [app.current_status || current_run()];
    return runs.filter(run => run && Number(run.width_snapshot_count || 0) > 0);
  }
  function ensure_group() {
    let group = by_id("width_chart_group");
    if (group) return group;
    group = document.createElement("section"); group.id = "width_chart_group"; group.className = "chart-group";
    const heading = document.createElement("h2"); heading.textContent = "Residual width";
    const grid = document.createElement("div"); grid.className = "chart-grid";
    chart_names.forEach((name, index) => {
      chart_titles[name] = titles[index];
      app.dynamic_chart_metadata[name] = {x_source:axis_labels[index][0], x_label:axis_labels[index][0],
        y_source:axis_labels[index][1], y_label:axis_labels[index][1], default_x_axis_mode:null, available_x_axis_modes:[]};
      const card = depth_card(name);
      card.querySelector("p").textContent = "Select a captured step and residual site.";
      const controls = document.createElement("div"); controls.style.cssText = "position:relative;z-index:3;display:flex;gap:4px;padding:6px 8px;flex-wrap:wrap;background:white;height:60px;align-content:flex-start";
      const step = document.createElement("select"); step.dataset.widthStep = name; step.title = "Captured optimizer update";
      const site = document.createElement("select"); site.dataset.widthSite = name; site.title = "Named residual capture site";
      step.style.cssText = site.style.cssText = "max-width:100%;font-size:10px";
      step.onchange = () => { selected_steps.set(name, step.value); rendered_signature = null; void refresh_width(); };
      site.onchange = () => { selected_sites.set(name, site.value); rendered_signature = null; void refresh_width(); };
      controls.append(step, site); card.querySelector("header").after(controls); grid.append(card);
      card.querySelector(".plot-shell").style.top = "112px";
    });
    group.append(heading, grid);
    by_id("charts_scroll").append(group);
    group.hidden = true;
    return group;
  }
  function fill_select(select, values, selected, prefix) {
    const signature = JSON.stringify(values);
    if (select.dataset.width_options !== signature) {
      select.replaceChildren(...values.map(value => { const option = document.createElement("option"); option.value = String(value); option.textContent = `${prefix}${value}`; return option; }));
      select.dataset.width_options = signature;
    }
    if (selected !== null && selected !== undefined) select.value = String(selected);
  }
  async function refresh_width() {
    const group = ensure_group(), runs = visible_runs();
    group.hidden = !runs.length;
    if (!runs.length || !instra_charts_visible() || document.hidden || by_id("charts_scroll")?.hidden) {
      owner?.controller.abort();
      return;
    }
    const context = app.workspace_mode ? window.__instra_workspace.selection_key() : `run:${app.current_run_id}`;
    const signature = JSON.stringify([context, runs.map(run => [run_identifier(run), run.width_maximum_update, colour_for_run(run_identifier(run))]), [...selected_steps], [...selected_sites]]);
    if (signature === rendered_signature && chart_names.every(name => {
      const card = by_id(`${name}_plot`).closest(".chart-card"), bounds = card.getBoundingClientRect(), viewport = by_id("charts_scroll").getBoundingClientRect();
      return bounds.bottom < viewport.top - 120 || bounds.top > viewport.bottom + 120 ||
        app.maximized_chart && app.maximized_chart !== name || by_id(`${name}_plot`).dataset.plotReady === "true" || card.dataset.widthEmpty === signature;
    })) return;
    if (owner) { if (owner.signature === signature) return owner.promise; owner.controller.abort(); }
    const request_owner = {signature, controller:new AbortController(), promise:null}; owner = request_owner;
    const timeout = setTimeout(() => request_owner.controller.abort(), 15000);
    request_owner.promise = (async () => {
      try {
        for (const chart_name of chart_names) {
          const card = by_id(`${chart_name}_plot`).closest(".chart-card"), bounds = card.getBoundingClientRect(), viewport = by_id("charts_scroll").getBoundingClientRect();
          if (app.maximized_chart && app.maximized_chart !== chart_name || bounds.bottom < viewport.top - 120 || bounds.top > viewport.bottom + 120) continue;
          const entries = (await window.__instra_workspace.map_with_concurrency(runs, 4, async run => {
            const query = new URLSearchParams({run:run_identifier(run), chart:chart_name});
            if (selected_steps.get(chart_name)) query.set("step", selected_steps.get(chart_name));
            if (selected_sites.get(chart_name)) query.set("site", selected_sites.get(chart_name));
            return {run, payload:await fetch_json(`/api/width-figure?${query}`, {signal:request_owner.controller.signal})};
          }, request_owner.controller.signal)).filter(Boolean);
          if (request_owner.controller.signal.aborted || owner !== request_owner) return;
          if (!entries.length) continue;
          const first = entries.find(entry => entry.payload.figure)?.payload || entries[0].payload;
          const steps = [...new Set(entries.flatMap(entry => entry.payload.available_steps || []))].sort((a, b) => a - b);
          const sites = [...new Set(entries.flatMap(entry => entry.payload.available_sites || []))];
          fill_select(card.querySelector("[data-width-step]"), steps, selected_steps.get(chart_name) || first.selected_step, "Step ");
          fill_select(card.querySelector("[data-width-site]"), sites, selected_sites.get(chart_name) || first.selected_site, "");
          card.querySelector("[data-width-site]").hidden = chart_name === "width_probes";
          const source = first.figure;
          const mount = by_id(`${chart_name}_plot`), placeholder = by_id(`${chart_name}_placeholder`);
          if (!source) { clear_plot(mount); card.dataset.widthEmpty = signature; placeholder.textContent = "No capture for this view at the selected step. Curves and loss ablations require a probe step."; placeholder.hidden = false; continue; }
          delete card.dataset.widthEmpty;
          const figure = {data:[], layout:structuredClone(source.layout)};
          for (const {run, payload} of entries) {
            for (const original_trace of payload.figure?.data || []) {
              const trace = structuredClone(original_trace), colour = colour_for_run(run_identifier(run));
              trace.name = `${run_name(run)} · ${trace.name}`; trace.legendgroup = run_identifier(run);
              trace.line = {...trace.line, color:colour}; trace.marker = {...trace.marker, color:colour};
              trace.meta = {...trace.meta, instra_workspace_run_id:run_identifier(run), instra_workspace_colour:colour};
              trace.hovertemplate = `Run: ${run_name(run)}<br>Step: ${payload.selected_step}<br>D=${payload.representation.reference_width}, r=${payload.representation.residual_width}<br>%{x}: %{y}<extra>%{fullData.name}</extra>`;
              figure.data.push(trace);
            }
          }
          app.dynamic_chart_figures[chart_name] = figure;
          by_id(`${chart_name}_detail`).textContent = `D=${first.representation.reference_width}, r=${first.representation.residual_width} · step ${first.selected_step} · ${first.selected_site || "trained-model sensitivity; positive delta loss means harm"}`;
          placeholder.hidden = true;
          await render_plot(mount, figure, chart_name);
        }
        rendered_signature = signature;
      } catch (error) { if (error.name !== "AbortError") console.warn("Width capture unavailable", error); }
      finally { clearTimeout(timeout); if (owner === request_owner) owner = null; }
    })();
    return request_owner.promise;
  }
  window.addEventListener("load", () => {
    ensure_group();
    by_id("charts_scroll").addEventListener("scroll", () => { rendered_signature = null; void refresh_width(); }, {passive:true});
    // Existing demand and controls owners finish installing at 400 ms after load.
    setTimeout(() => {
      const original_refresh = refresh_current_run;
      refresh_current_run = async function(...args) { const result = await original_refresh(...args); await refresh_width(); return result; };
      void refresh_width();
    }, 450);
  });
  window.instra_width = {refresh:refresh_width};
})();
// ^^^ THOG
