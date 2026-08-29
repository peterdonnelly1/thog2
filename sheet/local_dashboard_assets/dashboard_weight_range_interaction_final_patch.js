// vvv THOG
"use strict";

// Final owner for retained-range rendering and the top-level coupling viewer.
// Recorded couplings redraw immediately. A different in-bounds coupling can also be
// selected for a running run; it becomes available from the next retained snapshot
// because historical scalar trajectories cannot be reconstructed retroactively.
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
  const viewer_storage_key = "thog2_local_weight_viewer_couplings_v2";
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
    queueMicrotask(() => {
      if (document.activeElement?.matches?.("#weight_coupling_input, #weight_coupling_output")) return;
      viewer_controller?.commit_inputs?.();
    });
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
          const selection_kind = String(meta.instra_weight_selection_kind || "random");
          const pair = {model_feature, intermediate_feature, selection_kind};
          const key = pair_key(pair);
          const prior = pairs.get(key);
          if (!prior || (prior.selection_kind === "user" && selection_kind !== "user")) {
            pairs.set(key, pair);
          }
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

    const viewer_capability = () => {
      for (const chart_name of weight_chart_names) {
        try {
          const capability = capture_api.capability?.(chart_name);
          const maximum = finite_integer(capability?.maximum);
          if (capability?.available === true && maximum !== null && maximum >= 0) {
            return {available: true, maximum};
          }
        } catch (_error) {}
      }
      return {available: false, maximum: null};
    };

    const pair_in_bounds = pair => {
      const capability = viewer_capability();
      return Boolean(
        capability.available
        && pair
        && finite_integer(pair.model_feature) !== null
        && finite_integer(pair.intermediate_feature) !== null
        && pair.model_feature >= 0
        && pair.intermediate_feature >= 0
        && pair.model_feature <= capability.maximum
        && pair.intermediate_feature <= capability.maximum
      );
    };

    const selected_run_is_active = () => {
      const run = typeof current_run === "function" ? current_run() : app.current_status;
      let state = "";
      try { state = display_run_state(run); }
      catch (_error) { state = String(run?.run_state || ""); }
      return state === "preparing" || state === "recording" || state === "monitoring" || state === "running";
    };

    const viewer_pair = () => {
      const pairs = recorded_pairs();
      const stored = stored_pair();
      const captured = capture_selection();
      const captured_pair = captured.user_selected === true
        ? {
            model_feature: finite_integer(captured.model_feature),
            intermediate_feature: finite_integer(captured.intermediate_feature),
          }
        : null;
      const active = selected_run_is_active();
      const resolved = pairs.find(pair => same_pair(pair, stored))
        || (active && pair_in_bounds(stored) ? stored : null)
        || pairs.find(pair => same_pair(pair, captured_pair))
        || (active && pair_in_bounds(captured_pair) ? captured_pair : null)
        || pairs.find(pair => pair.selection_kind !== "user")
        || pairs[0];
      if (!resolved) return null;
      if (!same_pair(resolved, stored)) persist_pair(resolved);
      return {
        model_feature: resolved.model_feature,
        intermediate_feature: resolved.intermediate_feature,
      };
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
      return {
        protocol,
        user_selected: true,
        model_feature: pair.model_feature,
        intermediate_feature: pair.intermediate_feature,
      };
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

    const coupling_input_is_being_edited = () => (
      document.activeElement?.matches?.("#weight_coupling_input, #weight_coupling_output") === true
    );

    const sync_viewer_controls = () => {
      const pairs = recorded_pairs();
      const pair = viewer_pair();
      const capability = viewer_capability();
      const input = by_id("weight_coupling_input");
      const output = by_id("weight_coupling_output");
      if (!input || !output) return;
      if (pair && !coupling_input_is_being_edited()) {
        write_input(input, pair.model_feature);
        write_input(output, pair.intermediate_feature);
      }
      if (capability.maximum !== null) {
        input.max = String(capability.maximum);
        output.max = String(capability.maximum);
      }
      const disabled = !capability.available;
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
        if (button) button.disabled = disabled || capability.maximum < 1;
      }
      const button_titles = new Map([
        ["weight_residual_minus", "Select the preceding valid input feature"],
        ["weight_residual_plus", "Select the next valid input feature"],
        ["weight_branch_minus", "Select the preceding valid output feature"],
        ["weight_branch_plus", "Select the next valid output feature"],
        ["weight_random_jump", "Select a different random valid feature coupling"],
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
        editor.title = capability.available
          ? `Valid feature indices: 0–${capability.maximum}; ${pairs.length} coupling${pairs.length === 1 ? "" : "s"} recorded across every displayed step`
          : "Waiting for the run's matrix dimensions";
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

    const schedule_capture_pair = async candidate => {
      if (typeof capture_api.save !== "function") return false;
      if (!selected_run_is_active()) return false;
      await capture_api.save(candidate.model_feature, candidate.intermediate_feature);
      persist_pair(candidate);
      sync_viewer_controls();
      request_render();
      show_error(
        `Coupling ${candidate.model_feature} → ${candidate.intermediate_feature} is valid and will appear from the next recorded snapshot; earlier snapshots cannot be reconstructed.`
      );
      return true;
    };

    const reject_unrecorded_pair = candidate => {
      const current = viewer_pair();
      show_error(
        `Coupling ${candidate.model_feature} → ${candidate.intermediate_feature} was not recorded for this completed view.`
      );
      if (current) {
        write_input(by_id("weight_coupling_input"), current.model_feature);
        write_input(by_id("weight_coupling_output"), current.intermediate_feature);
      }
      return false;
    };

    const commit_pair = candidate => {
      if (select_pair(candidate)) return true;
      // Historical rejection is deliberately synchronous: the values and red
      // message must be corrected before the input event returns.
      if (!selected_run_is_active()) return reject_unrecorded_pair(candidate);
      return (async () => {
        try {
          if (await schedule_capture_pair(candidate)) return true;
        } catch (error) {
          show_error(`Weight coupling save failed: ${error.message}`);
          return false;
        }
        return reject_unrecorded_pair(candidate);
      })();
    };

    const commit_inputs = () => {
      const model_feature = finite_integer(by_id("weight_coupling_input")?.value);
      const intermediate_feature = finite_integer(by_id("weight_coupling_output")?.value);
      const current = viewer_pair();
      const capability = viewer_capability();
      if (model_feature === null || intermediate_feature === null || model_feature < 0 || intermediate_feature < 0) {
        show_error("Both coupling indices must be non-negative whole numbers.");
        sync_viewer_controls();
        return false;
      }
      if (
        !capability.available
        || model_feature > capability.maximum
        || intermediate_feature > capability.maximum
      ) {
        show_error(
          capability.available
            ? `Both coupling indices must be between 0 and ${capability.maximum}.`
            : "Waiting for this run's matrix dimensions."
        );
        if (current) {
          write_input(by_id("weight_coupling_input"), current.model_feature);
          write_input(by_id("weight_coupling_output"), current.intermediate_feature);
        }
        return false;
      }
      const candidate = {model_feature, intermediate_feature};
      void commit_pair(candidate);
      return true;
    };

    const directional_recorded_pair = (button_id, current, pairs) => {
      const input_axis = button_id.startsWith("weight_residual_");
      const increasing = button_id.endsWith("_plus");
      const axis = input_axis ? "model_feature" : "intermediate_feature";
      const other = input_axis ? "intermediate_feature" : "model_feature";
      return pairs
        .filter(pair => increasing ? pair[axis] > current[axis] : pair[axis] < current[axis])
        .sort((left, right) => (
          Number(right[other] === current[other]) - Number(left[other] === current[other])
          || Math.abs(left[axis] - current[axis]) - Math.abs(right[axis] - current[axis])
          || Math.abs(left[other] - current[other]) - Math.abs(right[other] - current[other])
        ))[0] || null;
    };

    const handle_button = button_id => {
      const current = viewer_pair();
      const capability = viewer_capability();
      if (!capability.available || !current) {
        show_error("Waiting for this run's matrix dimensions and first coupling.");
        sync_viewer_controls();
        return;
      }
      if (!selected_run_is_active()) {
        const recorded = recorded_pairs();
        if (button_id === "weight_random_jump") {
          const both_changed = recorded.filter(pair => (
            pair.model_feature !== current.model_feature
            && pair.intermediate_feature !== current.intermediate_feature
          ));
          const any_changed = recorded.filter(pair => !same_pair(pair, current));
          const choices = both_changed.length ? both_changed : any_changed;
          if (!choices.length) {
            show_error("No other recorded coupling is available for this completed view.");
            return;
          }
          void commit_pair(choices[Math.floor(Math.random() * choices.length)]);
          return;
        }
        const next_recorded = directional_recorded_pair(button_id, current, recorded);
        if (!next_recorded) {
          show_error("No recorded coupling is available in that direction for this completed view.");
          return;
        }
        void commit_pair(next_recorded);
        return;
      }
      if (button_id === "weight_random_jump") {
        const random_other = value => {
          if (capability.maximum < 1) return value;
          const draw = Math.floor(Math.random() * capability.maximum);
          return draw >= value ? draw + 1 : draw;
        };
        void commit_pair({
          model_feature: random_other(current.model_feature),
          intermediate_feature: random_other(current.intermediate_feature),
        });
        return;
      }
      const next = {...current};
      if (button_id === "weight_residual_minus") next.model_feature = Math.max(0, next.model_feature - 1);
      if (button_id === "weight_residual_plus") next.model_feature = Math.min(capability.maximum, next.model_feature + 1);
      if (button_id === "weight_branch_minus") next.intermediate_feature = Math.max(0, next.intermediate_feature - 1);
      if (button_id === "weight_branch_plus") next.intermediate_feature = Math.min(capability.maximum, next.intermediate_feature + 1);
      if (same_pair(next, current)) {
        show_error(`Feature index is already at ${button_id.endsWith("minus") ? 0 : capability.maximum}.`);
        return;
      }
      void commit_pair(next);
    };

    const viewer_api = {
      selection: viewer_selection,
      pair: viewer_pair,
      recorded_pairs: () => recorded_pairs().map(pair => ({
        model_feature: pair.model_feature,
        intermediate_feature: pair.intermediate_feature,
      })),
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

    const blend_rgb = (left, right, amount) => left.map((value, index) => (
      Math.round(value + (right[index] - value) * amount)
    ));

    const gradient_colour = (base_colour, update, minimum, maximum) => {
      const base = typeof hex_to_rgb === "function" ? hex_to_rgb(base_colour) : null;
      if (!base || maximum <= minimum) return base_colour;
      const position = Math.max(0, Math.min(1, (update - minimum) / (maximum - minimum)));
      if (position === 0.5) return base_colour;
      if (position < 0.5) {
        const lightest = blend_rgb(base, [255, 255, 255], 0.64);
        return rgb_to_hex(blend_rgb(lightest, base, position / 0.5));
      }
      const darkest = blend_rgb(base, [0, 0, 0], 0.42);
      return rgb_to_hex(blend_rgb(base, darkest, (position - 0.5) / 0.5));
    };

    const trace_run_id = trace => String(
      trace?.meta?.instra_workspace_run_id || app.current_run_id || "__instra_single_run__"
    );

    const redraw_mounted_weight_figures = async () => {
      if (typeof render_plot !== "function") return;
      const jobs = [];
      for (const chart_name of weight_chart_names) {
        const mount = by_id(`${chart_name}_plot`);
        const figure = mount?.__instraWeightFigure || app.figures?.depth?.[chart_name];
        if (!mount || !figure) continue;
        jobs.push(render_plot(mount, figure, chart_name));
      }
      await Promise.all(jobs);
      sync_viewer_controls();
    };

    const colour_weight_steps = prepared => {
      const use_gradient = stability.gradient_enabled?.() === true;
      if (app.workspace_mode === true && !use_gradient) return;
      if (use_gradient) {
        const steps_by_run = new Map();
        for (const trace of prepared?.data || []) {
          if (trace?.meta?.instra_top_axis_anchor === true) continue;
          let update = null;
          try { update = finite_integer(trace_optimizer_update(trace)); }
          catch (_error) { update = null; }
          if (update === null) continue;
          const identifier = trace_run_id(trace);
          if (!steps_by_run.has(identifier)) steps_by_run.set(identifier, new Set());
          steps_by_run.get(identifier).add(update);
        }
        const bounds_by_run = new Map([...steps_by_run].map(([identifier, values]) => {
          const steps = [...values].sort((left, right) => left - right);
          return [identifier, {
            count: steps.length,
            minimum: steps[0],
            maximum: steps[steps.length - 1],
          }];
        }));
        const colours = [];
        for (const trace of prepared?.data || []) {
          if (trace?.meta?.instra_top_axis_anchor === true) continue;
          let update = null;
          try { update = finite_integer(trace_optimizer_update(trace)); }
          catch (_error) { update = null; }
          if (update === null) continue;
          const identifier = trace_run_id(trace);
          const bounds = bounds_by_run.get(identifier);
          const base_colour = colour_for_run(identifier);
          const colour = !bounds || bounds.count <= 1
            ? base_colour
            : gradient_colour(base_colour, update, bounds.minimum, bounds.maximum);
          colours.push(colour);
          const mode = String(trace.mode || "");
          if (mode.includes("lines") || trace.line) trace.line = {...(trace.line || {}), color: colour};
          if (mode.includes("markers") || trace.marker) {
            trace.marker = {...(trace.marker || {}), color: colour};
            trace.marker.line = {...(trace.marker?.line || {}), color: colour};
          }
        }
        prepared.layout = {...(prepared.layout || {}), colorway: colours};
        return;
      }
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

    const escaped_html = value => String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

    const selected_run_artifact = () => {
      let run = null;
      try { run = typeof current_run === "function" ? current_run() : null; }
      catch (_error) { run = null; }
      return String(run?.artifact_name || run?.run_name || app.current_status?.artifact_name || "");
    };

    const artifact_datetime = artifact => {
      const value = String(artifact || "");
      const match = value.match(/^(\d{6}-\d{4}|\d{2}-\d{3,4}-\d{4})(?:_|$)/);
      return match ? match[1] : value;
    };

    const format_weight_hover = (prepared, chart_name) => {
      for (const trace of prepared?.data || []) {
        if (trace?.meta?.instra_top_axis_anchor === true) continue;
        let update = null;
        try { update = finite_integer(trace_optimizer_update(trace)); }
        catch (_error) { update = null; }
        if (update === null) continue;
        const existing = typeof trace.hovertemplate === "string" ? trace.hovertemplate : "";
        const extra_index = existing.indexOf("<extra");
        const body = extra_index >= 0 ? existing.slice(0, extra_index) : existing;
        const extra = extra_index >= 0 ? existing.slice(extra_index) : "<extra></extra>";
        let rows = body.split("<br>").filter(Boolean);
        const meta = trace?.meta && typeof trace.meta === "object" && !Array.isArray(trace.meta)
          ? trace.meta
          : {};
        const artifact = String(meta.instra_workspace_artifact_name || selected_run_artifact());
        const compact_identity = String(meta.instra_workspace_run_datetime || artifact_datetime(artifact));
        if (artifact && /^<b>[\s\S]*<\/b>$/.test(rows[0] || "")) rows = rows.slice(1);
        rows = rows.filter(row => (
          !new RegExp(`^step\\s+${update}\\s*$`, "i").test(row)
          && !new RegExp(`^U${update}(?:\\s*[·•].*)?$`, "i").test(row)
        ));
        const identity = app.maximized_chart === chart_name ? artifact : compact_identity;
        if (identity) {
          rows = [`<b>${escaped_html(identity)}</b>`, `step ${update}`, ...rows];
        } else if (rows.length) {
          rows = [rows[0], `step ${update}`, ...rows.slice(1)];
        } else {
          rows = [`step ${update}`];
        }
        trace.hovertemplate = `${rows.join("<br>")}${extra}`;
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
      const selected_pair = viewer_pair();
      if (selected_pair) {
        prepared.data = (prepared.data || []).filter(trace => {
          const meta = trace?.meta;
          if (!meta || typeof meta !== "object" || Array.isArray(meta)) return true;
          if (meta.instra_weight_selection_protocol !== protocol) return true;
          return (
            finite_integer(meta.instra_weight_model_feature) === selected_pair.model_feature
            && finite_integer(meta.instra_weight_intermediate_feature) === selected_pair.intermediate_feature
          );
        });
      }
      format_weight_hover(prepared, chart_name);
      colour_weight_steps(prepared);
      return prepared;
    };

    let hover_mode_render_queued = false;
    const queue_hover_mode_render = () => {
      if (hover_mode_render_queued) return;
      hover_mode_render_queued = true;
      queueMicrotask(() => {
        hover_mode_render_queued = false;
        try {
          const result = render_figures();
          if (result && typeof result.catch === "function") {
            result.catch(error => show_error(`Weight hover redraw failed: ${error.message}`));
          }
        } catch (error) {
          show_error(`Weight hover redraw failed: ${error.message}`);
        }
      });
    };

    if (typeof restore_maximized_chart === "function") {
      const base_restore_maximized_chart_weight_hover = restore_maximized_chart;
      restore_maximized_chart = function() {
        const prior = app.maximized_chart;
        const result = base_restore_maximized_chart_weight_hover();
        if (weight_chart_set.has(prior)) queue_hover_mode_render();
        return result;
      };
    }

    if (typeof toggle_maximized_chart === "function") {
      const base_toggle_maximized_chart_weight_hover = toggle_maximized_chart;
      toggle_maximized_chart = function(chart_name) {
        const prior = app.maximized_chart;
        const result = base_toggle_maximized_chart_weight_hover(chart_name);
        if (weight_chart_set.has(chart_name) || weight_chart_set.has(prior)) {
          queue_hover_mode_render();
        }
        return result;
      };
    }

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
      redraw_mounted: redraw_mounted_weight_figures,
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
