// vvv THOG
"use strict";

// Final owner for retained-range rendering and the top-level coupling viewer. The
// header coupling is deliberately display-only: historical snapshots contain a
// bounded set of scalar trajectories, so the viewer may select only a logical pair
// recorded for every chart/run/step currently on screen. Trainer capture remains in
// the chart settings and applies to future snapshots.
window.addEventListener("load", () => {
  const weight_chart_names = Object.freeze([
    "attn_q_head_N",
    "attn_k_head_N",
    "attn_v_head_N",
    "attn_out_head_N",
    "mlp_up",
    "mlp_down",
  ]);
  const weight_chart_set = new Set(weight_chart_names);
  const protocol = "matched_six_v1";
  const viewer_storage_key = "thog2_local_weight_viewer_couplings_v1";
  let viewer_controller = null;

  const finite_integer = value => {
    if (value === null || value === undefined || value === "") return null;
    const numeric = Number(value);
    return Number.isInteger(numeric) ? numeric : null;
  };

  const pair_key = pair => `${pair.model_feature}:${pair.intermediate_feature}`;
  const same_pair = (left, right) => Boolean(
    left
    && right
    && finite_integer(left.model_feature) === finite_integer(right.model_feature)
    && finite_integer(left.intermediate_feature) === finite_integer(right.intermediate_feature)
  );

  // These capture listeners are registered before the older delayed control owners.
  // They prevent a viewer action from POSTing a new trainer-capture selection.
  window.addEventListener("click", event => {
    const button = event.target?.closest?.([
      "#weight_residual_minus",
      "#weight_residual_plus",
      "#weight_branch_minus",
      "#weight_branch_plus",
      "#weight_random_jump",
    ].join(","));
    if (!button) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    viewer_controller?.handle_button?.(button.id);
  }, true);

  window.addEventListener("change", event => {
    if (!event.target?.matches?.("#weight_coupling_input, #weight_coupling_output")) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    viewer_controller?.commit_inputs?.();
  }, true);

  window.addEventListener("keydown", event => {
    if (event.key !== "Enter") return;
    if (!event.target?.matches?.("#weight_coupling_input, #weight_coupling_output")) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    viewer_controller?.commit_inputs?.();
  }, true);

  window.addEventListener("input", event => {
    if (!event.target?.matches?.("#weight_coupling_input, #weight_coupling_output")) return;
    viewer_controller?.show_error?.("");
  }, true);

  const install_final_render_owner = () => {
    if (window.__instra_weight_range_interaction_final) return true;
    const stability = window.__instra_weight_stability_final;
    const capture_api = window.__instra_matched_weight_selection;
    if (!stability || !capture_api) return false;
    if (!window.__instra_weight_coupling_reliability_final) return false;
    if (typeof prepare_figure !== "function" || typeof render_figures !== "function") return false;

    const context_key = () => String(stability.context_key?.() || "");
    const read_viewer_store = () => {
      const value = typeof load_json === "function"
        ? load_json(viewer_storage_key, {})
        : (() => {
            try { return JSON.parse(localStorage.getItem(viewer_storage_key) || "{}"); }
            catch (_error) { return {}; }
          })();
      return value && typeof value === "object" && !Array.isArray(value) ? value : {};
    };
    const write_viewer_store = value => {
      if (typeof save_json === "function") save_json(viewer_storage_key, value);
      else localStorage.setItem(viewer_storage_key, JSON.stringify(value));
    };
    const stored_pair = () => {
      const value = read_viewer_store()[context_key()];
      const model_feature = finite_integer(value?.model_feature);
      const intermediate_feature = finite_integer(value?.intermediate_feature);
      return model_feature === null || intermediate_feature === null
        ? null
        : {model_feature, intermediate_feature};
    };
    const persist_pair = pair => {
      const key = context_key();
      if (!key || !pair) return;
      const store = read_viewer_store();
      store[key] = {
        model_feature: pair.model_feature,
        intermediate_feature: pair.intermediate_feature,
      };
      write_viewer_store(store);
    };

    const raw_figure = chart_name => {
      try { return figure_for_chart(chart_name); }
      catch (_error) { return null; }
    };

    // Intersect logical pairs across every visible chart/run/optimizer-step unit.
    // A chosen pair therefore cannot make part of a multi-step view disappear.
    const recorded_pairs = () => {
      const pairs = new Map();
      const units = new Map();
      for (const chart_name of weight_chart_names) {
        const figure = raw_figure(chart_name);
        for (const trace of figure?.data || []) {
          const meta = trace?.meta;
          if (!meta || typeof meta !== "object" || Array.isArray(meta)) continue;
          if (meta.instra_weight_selection_protocol !== protocol) continue;
          if (meta.instra_top_axis_anchor === true || meta.instra_thog_executed_overlay === true) continue;
          const model_feature = finite_integer(meta.instra_weight_model_feature);
          const intermediate_feature = finite_integer(meta.instra_weight_intermediate_feature);
          let optimizer_update = null;
          try { optimizer_update = finite_integer(trace_optimizer_update(trace)); }
          catch (_error) { optimizer_update = null; }
          if (
            model_feature === null
            || intermediate_feature === null
            || model_feature < 0
            || intermediate_feature < 0
            || optimizer_update === null
          ) continue;
          const pair = {model_feature, intermediate_feature};
          const key = pair_key(pair);
          pairs.set(key, pair);
          const run_id = String(meta.instra_workspace_run_id || "__instra_single_run__");
          const unit_key = `${chart_name}:${run_id}:${optimizer_update}`;
          if (!units.has(unit_key)) units.set(unit_key, new Set());
          units.get(unit_key).add(key);
        }
      }
      const sets = [...units.values()];
      if (!sets.length) return [];
      return [...sets[0]]
        .filter(key => sets.every(unit => unit.has(key)))
        .map(key => pairs.get(key))
        .filter(Boolean)
        .sort((left, right) => (
          left.model_feature - right.model_feature
          || left.intermediate_feature - right.intermediate_feature
        ));
    };

    const capture_selection = () => {
      const value = capture_api.selection?.();
      return value && typeof value === "object" ? value : {};
    };

    const viewer_pair = () => {
      const pairs = recorded_pairs();
      if (!pairs.length) return null;
      const stored = stored_pair();
      const captured = capture_selection();
      const resolved = pairs.find(pair => same_pair(pair, stored))
        || pairs.find(pair => same_pair(pair, captured))
        || pairs[0];
      if (!same_pair(resolved, stored)) persist_pair(resolved);
      return {...resolved};
    };

    const viewer_selection = () => {
      const pair = viewer_pair();
      if (!pair) {
        return {
          protocol,
          user_selected: false,
          model_feature: finite_integer(capture_selection().model_feature) ?? 0,
          intermediate_feature: finite_integer(capture_selection().intermediate_feature) ?? 0,
        };
      }
      return {protocol, user_selected: true, ...pair};
    };

    const ensure_error = () => {
      const editor = by_id("weight_coupling_editor");
      if (!editor) return null;
      let error = by_id("weight_coupling_view_error");
      if (!error) {
        error = document.createElement("span");
        error.id = "weight_coupling_view_error";
        error.className = "weight-coupling-view-error";
        error.setAttribute("role", "status");
        error.hidden = true;
        editor.insertAdjacentElement("afterend", error);
      }
      return error;
    };

    const show_error = message => {
      const error = ensure_error();
      if (!error) return;
      error.textContent = String(message || "");
      error.hidden = !message;
    };

    const write_input = (input, value) => {
      if (!input) return;
      const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
      if (descriptor?.set) descriptor.set.call(input, String(value));
      else input.value = String(value);
    };

    const sync_viewer_controls = () => {
      const pairs = recorded_pairs();
      const pair = viewer_pair();
      const input = by_id("weight_coupling_input");
      const output = by_id("weight_coupling_output");
      if (!input || !output) return;
      if (pair) {
        write_input(input, pair.model_feature);
        write_input(output, pair.intermediate_feature);
      }
      const disabled = !pairs.length;
      input.disabled = disabled;
      output.disabled = disabled;
      for (const id of [
        "weight_residual_minus",
        "weight_residual_plus",
        "weight_branch_minus",
        "weight_branch_plus",
        "weight_random_jump",
      ]) {
        const button = by_id(id);
        if (button) button.disabled = disabled || (id === "weight_random_jump" && pairs.length < 2);
      }
      const button_titles = new Map([
        ["weight_residual_minus", "Show the nearest recorded coupling with a lower input index"],
        ["weight_residual_plus", "Show the nearest recorded coupling with a higher input index"],
        ["weight_branch_minus", "Show the nearest recorded coupling with a lower output index"],
        ["weight_branch_plus", "Show the nearest recorded coupling with a higher output index"],
        ["weight_random_jump", "Show another coupling recorded for every displayed step"],
      ]);
      for (const [id, title] of button_titles) {
        const button = by_id(id);
        if (!button) continue;
        button.title = title;
        button.setAttribute("aria-label", title);
      }
      const random = by_id("weight_random_jump");
      if (random) random.textContent = "RND";
      const editor = by_id("weight_coupling_editor");
      if (editor) {
        editor.title = pairs.length
          ? `${pairs.length} coupling${pairs.length === 1 ? "" : "s"} recorded for every displayed step`
          : "No common recorded coupling is available for every displayed step";
      }
      ensure_error();
    };

    const request_render = () => {
      try {
        const result = render_figures();
        if (result && typeof result.catch === "function") {
          result.catch(error => show_error(`Weight coupling redraw failed: ${error.message}`));
        }
      } catch (error) {
        show_error(`Weight coupling redraw failed: ${error.message}`);
      }
    };

    const select_pair = (pair, {render = true} = {}) => {
      const available = recorded_pairs();
      const resolved = available.find(candidate => same_pair(candidate, pair));
      if (!resolved) return false;
      const changed = !same_pair(viewer_pair(), resolved);
      persist_pair(resolved);
      show_error("");
      sync_viewer_controls();
      if (changed && render) request_render();
      return true;
    };

    const commit_inputs = () => {
      const model_feature = finite_integer(by_id("weight_coupling_input")?.value);
      const intermediate_feature = finite_integer(by_id("weight_coupling_output")?.value);
      const current = viewer_pair();
      if (model_feature === null || intermediate_feature === null || model_feature < 0 || intermediate_feature < 0) {
        show_error("Both coupling indices must be non-negative whole numbers.");
        sync_viewer_controls();
        return false;
      }
      const candidate = {model_feature, intermediate_feature};
      if (select_pair(candidate)) return true;
      const range = stability.selected_range?.();
      const suffix = range
        ? ` in steps ${range.minimum}–${range.maximum}`
        : " in this view";
      show_error(`Coupling ${model_feature} → ${intermediate_feature} was not recorded for every displayed step${suffix}.`);
      if (current) {
        write_input(by_id("weight_coupling_input"), current.model_feature);
        write_input(by_id("weight_coupling_output"), current.intermediate_feature);
      }
      return false;
    };

    const directional_pair = (button_id, current, pairs) => {
      const input_axis = button_id.startsWith("weight_residual_");
      const increasing = button_id.endsWith("_plus");
      const axis = input_axis ? "model_feature" : "intermediate_feature";
      const other = input_axis ? "intermediate_feature" : "model_feature";
      const candidates = pairs.filter(pair => (
        increasing ? pair[axis] > current[axis] : pair[axis] < current[axis]
      ));
      candidates.sort((left, right) => (
        Number(right[other] === current[other]) - Number(left[other] === current[other])
        || Math.abs(left[axis] - current[axis]) - Math.abs(right[axis] - current[axis])
        || Math.abs(left[other] - current[other]) - Math.abs(right[other] - current[other])
      ));
      return candidates[0] || null;
    };

    const handle_button = button_id => {
      const pairs = recorded_pairs();
      const current = viewer_pair();
      if (!pairs.length || !current) {
        show_error("No common recorded coupling is available for these steps.");
        sync_viewer_controls();
        return;
      }
      if (button_id === "weight_random_jump") {
        let alternatives = pairs.filter(pair => (
          !same_pair(pair, current)
          && pair.model_feature !== current.model_feature
          && pair.intermediate_feature !== current.intermediate_feature
        ));
        if (!alternatives.length) alternatives = pairs.filter(pair => !same_pair(pair, current));
        if (!alternatives.length) {
          show_error("Only one coupling was recorded for every displayed step.");
          sync_viewer_controls();
          return;
        }
        select_pair(alternatives[Math.floor(Math.random() * alternatives.length)]);
        return;
      }
      const next = directional_pair(button_id, current, pairs);
      if (!next) {
        show_error("No recorded coupling is available in that direction for these steps.");
        sync_viewer_controls();
        return;
      }
      select_pair(next);
    };

    const viewer_api = {
      selection: viewer_selection,
      pair: viewer_pair,
      recorded_pairs: () => recorded_pairs().map(pair => ({...pair})),
      select_pair: (model_feature, intermediate_feature) => select_pair({
        model_feature: finite_integer(model_feature),
        intermediate_feature: finite_integer(intermediate_feature),
      }),
      sync: sync_viewer_controls,
    };
    window.__instra_weight_viewer_selection = viewer_api;
    viewer_controller = {commit_inputs, handle_button, show_error};

    const step_colour = update => {
      const hue = ((Number(update) * 137.50776405) % 360 + 360) % 360;
      return `hsl(${hue.toFixed(2)} 72% 46%)`;
    };

    const colour_weight_steps = prepared => {
      if (app.workspace_mode === true) return;
      const steps = [...new Set((prepared?.data || []).map(trace => {
        try { return finite_integer(trace_optimizer_update(trace)); }
        catch (_error) { return null; }
      }).filter(step => step !== null))];
      if (!steps.length) return;
      const single_colour = steps.length === 1
        ? colour_for_run(String(app.current_run_id || ""))
        : null;
      for (const trace of prepared.data || []) {
        if (trace?.meta?.instra_top_axis_anchor === true) continue;
        let update = null;
        try { update = finite_integer(trace_optimizer_update(trace)); }
        catch (_error) { update = null; }
        if (update === null) continue;
        const colour = single_colour || step_colour(update);
        const mode = String(trace.mode || "");
        if (mode.includes("lines") || trace.line) trace.line = {...(trace.line || {}), color: colour};
        if (mode.includes("markers") || trace.marker) {
          trace.marker = {...(trace.marker || {}), color: colour};
          trace.marker.line = {...(trace.marker?.line || {}), color: colour};
        }
      }
      prepared.layout = {...(prepared.layout || {}), colorway: steps.map(step => single_colour || step_colour(step))};
    };

    const add_step_hover = prepared => {
      for (const trace of prepared?.data || []) {
        if (trace?.meta?.instra_top_axis_anchor === true) continue;
        let update = null;
        try { update = finite_integer(trace_optimizer_update(trace)); }
        catch (_error) { update = null; }
        if (update === null) continue;
        const marker = `step ${update}`;
        const existing = typeof trace.hovertemplate === "string" ? trace.hovertemplate : "";
        if (existing.includes(marker)) continue;
        if (!existing) {
          trace.hovertemplate = `${marker}<br>layer %{x}<br>weight %{y}<extra></extra>`;
          continue;
        }
        const extra_index = existing.indexOf("<extra");
        trace.hovertemplate = extra_index >= 0
          ? `${existing.slice(0, extra_index)}<br>${marker}${existing.slice(extra_index)}`
          : `${existing}<br>${marker}`;
      }
    };

    const base_prepare_figure_weight_range_final = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      if (!weight_chart_set.has(chart_name)) {
        return base_prepare_figure_weight_range_final(figure, chart_name);
      }

      const range = stability.selected_range?.() || null;
      const saved_override = app.chart_settings_render_override;
      const preview = Boolean(
        by_id("chart_settings_overlay")?.hidden === false
        && saved_override?.chart_name === chart_name
        && saved_override.settings
      );
      if (range && !preview) {
        const supplied = saved_override?.chart_name === chart_name && saved_override.settings
          ? saved_override.settings
          : normalize_chart_settings(chart_name);
        app.chart_settings_render_override = {
          chart_name,
          settings: {...supplied, current_weights_only: false},
        };
      }

      let prepared;
      try {
        prepared = base_prepare_figure_weight_range_final(figure, chart_name);
      } finally {
        app.chart_settings_render_override = saved_override;
      }
      add_step_hover(prepared);
      colour_weight_steps(prepared);
      return prepared;
    };

    const base_render_figures_weight_range_final = render_figures;
    render_figures = async function() {
      try { return await base_render_figures_weight_range_final(); }
      finally { sync_viewer_controls(); }
    };

    if (typeof render_run_heading === "function") {
      const base_render_run_heading_weight_range_final = render_run_heading;
      render_run_heading = function() {
        const result = base_render_run_heading_weight_range_final();
        queueMicrotask(sync_viewer_controls);
        return result;
      };
    }

    if (typeof render_runs === "function") {
      const base_render_runs_weight_range_final = render_runs;
      render_runs = function() {
        const result = base_render_runs_weight_range_final();
        stability.sync_header?.();
        queueMicrotask(sync_viewer_controls);
        return result;
      };
    }

    const style = document.createElement("style");
    style.id = "thog2_weight_range_interaction_final_style";
    style.textContent = `
      .weight-coupling-editor input {
        width: 62px !important;
        min-width: 62px !important;
      }
      .weight-coupling-view-error {
        color: #b42318;
        font-size: 10px;
        font-weight: 400;
        line-height: 1.2;
        margin-left: 5px;
        white-space: nowrap;
      }
    `;
    document.head.appendChild(style);

    window.__instra_weight_range_interaction_final = Object.freeze({
      installed: true,
      viewer: viewer_api,
    });
    sync_viewer_controls();
    // The oldest header owner installs at 1.55 s and can briefly restore capture-
    // oriented values/enabled states. Reassert the viewer contract through that
    // bounded startup window; normal render wrappers own synchronization afterward.
    let startup_sync_passes = 0;
    const startup_sync_timer = setInterval(() => {
      startup_sync_passes += 1;
      stability.sync_header?.();
      sync_viewer_controls();
      if (startup_sync_passes >= 24) clearInterval(startup_sync_timer);
    }, 100);
    if (app.current_run_id) queueMicrotask(request_render);
    return true;
  };

  let attempts = 0;
  const retry = () => {
    attempts += 1;
    if (install_final_render_owner() || attempts >= 240) return;
    setTimeout(retry, 25);
  };
  // The regression guard intentionally installs at 360 ms; this outermost owner
  // waits until that full stack exists before taking final render responsibility.
  setTimeout(retry, 400);
});
// ^^^ THOG
