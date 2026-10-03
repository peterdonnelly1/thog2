// vvv THOG
"use strict";

/*
 * Workspace controller only.  This deliberately avoids the presentation and
 * Runs-table rewrites that lived in the older v0.58 repair bundle.
 */
window.addEventListener("load", () => {
  setTimeout(() => {
    const clone = value => JSON.parse(JSON.stringify(value));
    const finite_number = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };
    const run_name = run => String(run?.artifact_name || run?.run_name || run_identifier(run));
    const visible_runs = () => (app.runs || []).filter(run => is_visible(run_identifier(run)));
    let request_controller = null, request_membership = null;
    const cancel_pending = () => { request_controller?.abort(); request_controller=null; request_membership=null; };
    const workspace_request_signal = () => {
      const membership=visible_runs().map(run=>`${run_identifier(run)}:${colour_for_run(run_identifier(run))}`).sort().join("|");
      if(!request_controller || membership!==request_membership) {
        cancel_pending();request_controller=new AbortController();request_membership=membership;
      }
      return request_controller.signal;
    };
    const direct_json = async (url, signal) => {
      if(!signal)return fetch_json(url);
      const abort=new AbortController(), cancel=()=>abort.abort();
      if(signal.aborted)abort.abort();else signal.addEventListener("abort",cancel,{once:true});
      const deadline=setTimeout(cancel,30000);
      try{return await fetch_json(url,{signal:abort.signal});}
      finally{clearTimeout(deadline);signal.removeEventListener("abort",cancel);}
    };
    const map_with_concurrency = async (values, limit, operation, signal) => {
      const output = new Array(values.length);
      let cursor = 0;
      const worker = async () => {
        while (!signal?.aborted && cursor < values.length) {
          const index = cursor++;
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

    const weight_chart_names = Object.freeze([
      "attn_q_head_N", "attn_k_head_N", "attn_v_head_N",
      "attn_out_head_N", "mlp_up", "mlp_down",
    ]);

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
            const optimizer_update = source_optimizer_update(trace);
            trace.meta = {
              ...prior_meta,
              instra_workspace_run_id: id,
              instra_workspace_colour: colour,
              instra_workspace_optimizer_update: optimizer_update,
              instra_workspace_artifact_name: run_name(entry.run),
            };
            trace.name = `${run_name(entry.run)} · ${String(trace.name || chart_titles[chart_name] || chart_name)}`;
            trace.legendgroup = `instra-workspace-${id}`;
            trace.showlegend = false;
            if (trace.line) trace.line = {...trace.line, color: colour};
            if (trace.marker) trace.marker = {...trace.marker, color: colour};
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
      const ranges = entries.map(entry => entry.payload?.weight_step_range).filter(Boolean);
      return {
        depth,
        weight_step_range: ranges.length ? {
          minimum: ranges[0].minimum,
          maximum: ranges[0].maximum,
          snapshot_count: ranges.reduce((count, range) => count + Number(range.snapshot_count || 0), 0),
        } : null,
      };
    };

    const fetch_depth_payload = async request => {
      const runs = visible_runs().filter(run => Number(run.depth_snapshot_count || 0) > 0);
      const entries = await map_with_concurrency(runs, 8, async run => ({
        run,
        payload: await request(`/api/figure-family?run=${encodeURIComponent(run_identifier(run))}&family=depth`),
      }));
      return merge_depth_payloads(entries.filter(Boolean));
    };

    const fetch_metric_groups = async () => {
      const signal=workspace_request_signal();
      const entries = await map_with_concurrency(visible_runs(), 8, async run => ({
        run,
        payload: await direct_json(`/api/chart-groups?run=${encodeURIComponent(run_identifier(run))}`,signal),
      }),signal);
      if(signal?.aborted)throw Object.assign(new Error("Obsolete chart selection"),{name:"AbortError"});
      const groups = new Map();
      for (let run_index = 0; run_index < entries.length; run_index += 1) {
        const entry = entries[run_index];
        if (!entry?.payload?.available) continue;
        for (const summary of entry.payload.groups || []) {
          if (summary.name === "depth") continue;
          const current = groups.get(summary.name) || {name: summary.name, chart_count: 0, revision: 0};
          current.chart_count = Math.max(current.chart_count, Number(summary.chart_count || 0));
          current.revision += (run_index + 1) * Number(summary.revision || 0);
          groups.set(summary.name, current);
        }
      }
      return {available: true, source: "visible Instra runs", groups: [...groups.values()]};
    };

    const intersect_modes = (left, right) => {
      if (left === null) return [...right];
      const wanted = new Set(right);
      return left.filter(value => wanted.has(value));
    };

    const merge_metric_entries = (group_name, entries) => {
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
          // Both ordinary and Runner Multiview provide the native throughput
          // card; the historical duplicate must not displace loss.
          if (group_name === "train" &&
              /(?:token.*(?:sec|throughput)|throughput)/i.test(`${chart.id} ${chart.title}`)) continue;
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
      return {available: true, source: "visible Instra runs", group: {name: group_name, revision, charts: [...charts.values()]}};
    };

    const fetch_metric_group = async (group_name, on_progress) => {
      const signal=workspace_request_signal(), runs=visible_runs(), entries=new Array(runs.length);
      let last_progress=-Infinity, progress=Promise.resolve();
      await map_with_concurrency(runs,8,async(run,index)=>{
        const payload=await direct_json(`/api/chart-group?run=${encodeURIComponent(run_identifier(run))}&group=${encodeURIComponent(group_name)}`,signal);
        entries[index]={run,payload};
        if(on_progress && !signal?.aborted && Date.now()-last_progress>=300) {
          last_progress=Date.now();
          // vvv THOG one provisional first-result render; render the final merged set once all sources resolve
          const snapshot=merge_metric_entries(group_name,entries);
          if (last_progress !== Infinity) {
            last_progress=Infinity;
            progress=progress.then(()=>!signal?.aborted && on_progress(snapshot));
          }
          // ^^^ THOG
        }
      },signal);
      await progress;
      if(signal?.aborted)throw Object.assign(new Error("Obsolete chart selection"),{name:"AbortError"});
      return merge_metric_entries(group_name,entries);
    };

    let refresh_timer = null;
    let last_selection_key = "";
    const selection_key = () => visible_runs().map(run => {
      const id = run_identifier(run);
      return `${id}:${JSON.stringify(run.revision || [])}:${colour_for_run(id)}`;
    }).sort().join("|");

    app.workspace_mode = false;
    window.__instra_workspace = {
      active: () => app.workspace_mode === true,
      selection_key: () => `workspace:${selection_key()}`,
      visible_runs,
      fetch_depth_payload,
      fetch_metric_groups,
      fetch_metric_group,
      cancel_pending,
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
      clearTimeout(refresh_timer);
      refresh_timer = setTimeout(() => {
        app.figure_revision = null;
        if (app.figures) app.figures.depth = {};
        window.__thog2_metric_groups?.invalidate?.();
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
      // An untouched visibility map meant every historical run was selected.
      // Start with the current Grid; explicit saved eye choices remain authoritative.
      if(!Object.keys(app.visibility).length && app.runs?.length) {
        const selected=app.runs.find(run=>run_identifier(run)===app.current_run_id) || app.runs[0];
        for(const run of app.runs)app.visibility[run_identifier(run)]=selected.runner_grid_tag
          ? grid_identity(run)===grid_identity(selected) : run_identifier(run)===run_identifier(selected);
        save_json("thog2_local_run_visibility",app.visibility);render_runs();
      }
      const runs = visible_runs();
      if (!app.current_run_id) {
        const first = runs[0] || app.runs?.[0];
        if (first) select_run(run_identifier(first), {manual: false, replace_history: true});
      }
      app.workspace_mode = true;
      app.instra_loss_autofocused = false;
      window.__thog2_metric_groups?.clear?.();
      document.body.classList.add("instra-workspace-mode");
      by_id("workspace_nav")?.classList.add("selected");
      by_id("runs_nav")?.classList.remove("selected");
      by_id("settings_nav")?.classList.remove("selected");
      if (typeof local_set_detail_tab === "function") local_set_detail_tab("charts");
      last_selection_key = selection_key();
      render_workspace_heading();
      request_workspace_refresh();
      if (typeof processing_refresh === "function") processing_refresh(true);
    };

    const leave_workspace = () => {
      if (!app.workspace_mode) return;
      app.workspace_mode = false;
      cancel_pending();
      clearTimeout(refresh_timer);
      app.instra_loss_autofocused = false;
      app.instra_focus_loss = true;
      document.body.classList.remove("instra-workspace-mode");
      by_id("workspace_nav")?.classList.remove("selected");
      by_id("runs_nav")?.classList.add("selected");
      window.__thog2_metric_groups?.clear?.();
      app.figures = null;
      app.figure_revision = null;
      reset_run_charts();
      render_run_heading();
      refresh_current_run();
      if (typeof processing_refresh === "function") processing_refresh(true);
    };

    by_id("workspace_nav")?.addEventListener("click", enter_workspace);
    by_id("runs_nav")?.addEventListener("click", leave_workspace);

    const base_select_run = select_run;
    select_run = function(run_id, options = {}) {
      const result = base_select_run(run_id, options);
      if (app.workspace_mode) render_workspace_heading();
      return result;
    };

    const base_render_run_heading = render_run_heading;
    render_run_heading = function() {
      base_render_run_heading();
      render_workspace_heading();
    };

    const base_render_runs = render_runs;
    render_runs = function() {
      const result = base_render_runs();
      if (app.workspace_mode) {
        const next = selection_key();
        if (next !== last_selection_key) {
          last_selection_key = next;
          request_workspace_refresh();
        }
        render_workspace_heading();
      }
      return result;
    };

    setInterval(() => {
      if (!app.workspace_mode) return;
      const next = selection_key();
      if (next === last_selection_key) return;
      last_selection_key = next;
      request_workspace_refresh();
    }, 500);
  }, 0);
});
// ^^^ THOG
