// vvv THOG
"use strict";

// Final owner for one logical weight coordinate shared by all six weight charts.
// The capture selection is persisted per run so a coupling from one matrix shape
// cannot leak into another run. It affects future snapshots after it is saved.
window.addEventListener("load", () => {
  setTimeout(() => {
    const protocol = "matched_six_v1";
    const weight_chart_names = Object.freeze([
      "attn_q_head_N",
      "attn_k_head_N",
      "attn_v_head_N",
      "attn_out_head_N",
      "mlp_up",
      "mlp_down",
    ]);
    const weight_chart_set = new Set(weight_chart_names);
    const default_selection = Object.freeze({
      protocol,
      user_selected: true,
      model_feature: 0,
      intermediate_feature: 0,
    });

    let saved_selection = {...default_selection};
    let selection_loaded = false;
    let save_bypass = false;
    let save_in_flight = false;
    let selection_request_serial = 0;
    let loaded_run_id = "";

    const finite_integer = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isInteger(numeric) ? numeric : null;
    };

    const normalise_selection = supplied => {
      const source = supplied && typeof supplied === "object" && !Array.isArray(supplied)
        ? supplied
        : {};
      return {
        protocol,
        user_selected: source.user_selected === true,
        model_feature: finite_integer(source.model_feature) ?? 0,
        intermediate_feature: finite_integer(source.intermediate_feature) ?? 0,
      };
    };

    const same_selection = (left, right) => (
      left.user_selected === right.user_selected
      && left.model_feature === right.model_feature
      && left.intermediate_feature === right.intermediate_feature
    );

    const install_fields = () => {
      const current_field = by_id("chart_current_weights_only_field");
      if (!current_field || by_id("chart_matched_weight_section")) return;

      const section = document.createElement("section");
      section.id = "chart_matched_weight_section";
      section.className = "matched-weight-section";
      section.hidden = true;
      section.innerHTML = `
        <label class="chart-toggle-row matched-weight-toggle" for="chart_user_selected_weight">
          <span>
            <strong>User selected weight</strong>
            <small>Use one logical weight coordinate across all six weight matrices.</small>
          </span>
          <input id="chart_user_selected_weight" type="checkbox">
        </label>
        <div class="matched-weight-coordinate-grid">
          <label class="chart-editor-field" for="chart_weight_model_feature">
            <span>Model feature</span>
            <input id="chart_weight_model_feature" type="number" min="0" step="1" inputmode="numeric">
          </label>
          <label class="chart-editor-field" for="chart_weight_intermediate_feature">
            <span>Intermediate feature</span>
            <input id="chart_weight_intermediate_feature" type="number" min="0" step="1" inputmode="numeric">
          </label>
        </div>
        <p class="matched-weight-range" id="chart_matched_weight_range">Waiting for matrix dimensions…</p>
        <p class="matched-weight-note">
          0-based indices. Q/K/V and MLP expansion use intermediate ← model;
          attention output and MLP contraction use the reverse mapping.
        </p>
      `;
      current_field.insertAdjacentElement("afterend", section);

      for (const id of [
        "chart_user_selected_weight",
        "chart_weight_model_feature",
        "chart_weight_intermediate_feature",
      ]) {
        const control = by_id(id);
        control?.addEventListener("input", update_matched_weight_controls);
        control?.addEventListener("change", update_matched_weight_controls);
      }
      by_id("chart_current_weights_only")?.addEventListener("input", update_matched_weight_controls);
      by_id("chart_current_weights_only")?.addEventListener("change", update_matched_weight_controls);
    };

    const raw_figure_for_chart = chart_name => {
      try {
        return figure_for_chart(chart_name);
      } catch (_error) {
        return null;
      }
    };

    const protocol_trace_info = trace => {
      const meta = trace?.meta;
      if (!meta || typeof meta !== "object" || Array.isArray(meta)) return null;
      if (meta.instra_weight_selection_protocol !== protocol) return null;
      const feature_count = finite_integer(meta.instra_weight_feature_count);
      if (feature_count === null || feature_count < 1) return null;
      return {
        feature_count,
        run_id: meta.instra_workspace_run_id ? String(meta.instra_workspace_run_id) : null,
      };
    };

    const configured_feature_count = run => {
      const configuration = run?.configuration || {};
      for (const key of ["n_embd", "d_model", "model_width", "embedding_width"]) {
        const value = finite_integer(configuration[key]);
        if (value !== null && value > 0) return value;
      }
      return null;
    };

    const matched_weight_capability = chart_name => {
      if (!weight_chart_set.has(chart_name)) {
        return {available: false, maximum: null, reason: "Not a weight chart."};
      }
      const figure = raw_figure_for_chart(chart_name);
      const trace_info = (figure?.data || []).map(protocol_trace_info).filter(Boolean);
      if (!trace_info.length) {
        if (app.workspace_mode !== true) {
          const run = app.current_status || current_run();
          const feature_count = configured_feature_count(run);
          if (feature_count !== null) {
            return {available: true, maximum: feature_count - 1, reason: ""};
          }
        }
        return {
          available: false,
          maximum: null,
          reason: "Waiting for INSTRA to receive compatible weight-matrix dimensions.",
        };
      }

      if (app.workspace_mode === true) {
        const visible = typeof window.__instra_workspace?.visible_runs === "function"
          ? window.__instra_workspace.visible_runs()
          : [];
        const expected = visible
          .filter(run => Number(run?.depth_snapshot_count || 0) > 0)
          .map(run => String(run_identifier(run)));
        const by_run = new Map();
        for (const info of trace_info) {
          if (!info.run_id) continue;
          const prior = by_run.get(info.run_id);
          by_run.set(
            info.run_id,
            prior === undefined ? info.feature_count : Math.min(prior, info.feature_count),
          );
        }
        const missing = expected.filter(run_id => !by_run.has(run_id));
        if (missing.length) {
          return {
            available: false,
            maximum: null,
            reason: "Waiting for every visible weight run to report compatible matrix dimensions.",
          };
        }
        const counts = expected.map(run_id => by_run.get(run_id)).filter(Number.isFinite);
        if (!counts.length) {
          return {
            available: false,
            maximum: null,
            reason: "Waiting for INSTRA to receive compatible weight-matrix dimensions.",
          };
        }
        return {available: true, maximum: Math.min(...counts) - 1, reason: ""};
      }

      return {
        available: true,
        maximum: Math.min(...trace_info.map(info => info.feature_count)) - 1,
        reason: "",
      };
    };

    const selection_from_controls = () => normalise_selection({
      user_selected: by_id("chart_user_selected_weight")?.checked === true,
      model_feature: by_id("chart_weight_model_feature")?.value,
      intermediate_feature: by_id("chart_weight_intermediate_feature")?.value,
    });

    const resolved_current_only = chart_name => {
      if (!weight_chart_set.has(chart_name)) return false;
      if (app.axis_chart_name === chart_name && !by_id("chart_settings_overlay")?.hidden) {
        return by_id("chart_current_weights_only")?.checked === true;
      }
      const render_override = app.chart_settings_render_override;
      const settings = render_override?.chart_name === chart_name
        ? render_override.settings
        : normalize_chart_settings(chart_name);
      return settings?.current_weights_only === true;
    };

    function update_matched_weight_controls() {
      const section = by_id("chart_matched_weight_section");
      if (!section) return;
      const chart_name = app.axis_chart_name;
      const weight_chart = weight_chart_set.has(chart_name);
      section.hidden = !weight_chart;
      if (!weight_chart) return;

      const capability = matched_weight_capability(chart_name);
      const current_only = by_id("chart_current_weights_only")?.checked === true;
      const user_toggle = by_id("chart_user_selected_weight");
      const model_input = by_id("chart_weight_model_feature");
      const intermediate_input = by_id("chart_weight_intermediate_feature");
      const range = by_id("chart_matched_weight_range");
      const enabled = selection_loaded && capability.available && current_only;

      user_toggle.disabled = !enabled;
      model_input.disabled = !enabled || !user_toggle.checked;
      intermediate_input.disabled = !enabled || !user_toggle.checked;

      if (capability.available) {
        model_input.max = String(capability.maximum);
        intermediate_input.max = String(capability.maximum);
      } else {
        model_input.removeAttribute("max");
        intermediate_input.removeAttribute("max");
      }

      const model_feature = finite_integer(model_input.value);
      const intermediate_feature = finite_integer(intermediate_input.value);
      const out_of_range = capability.available && user_toggle.checked && (
        model_feature === null
        || intermediate_feature === null
        || model_feature < 0
        || intermediate_feature < 0
        || model_feature > capability.maximum
        || intermediate_feature > capability.maximum
      );
      section.classList.toggle("unavailable", !enabled);
      section.classList.toggle("invalid", out_of_range);

      if (!selection_loaded) {
        range.textContent = "Loading the persisted matched-weight selection…";
      } else if (!current_only) {
        range.textContent = "Select Current weights only to use a matched weight.";
      } else if (!capability.available) {
        range.textContent = capability.reason;
      } else if (out_of_range) {
        range.textContent = `Invalid coordinate. Both feature indices must be 0–${capability.maximum}.`;
      } else {
        range.textContent = (
          `Valid common range: 0–${capability.maximum} for both feature indices.`
        );
      }
    }

    const write_selection_controls = () => {
      if (!by_id("chart_user_selected_weight")) return;
      by_id("chart_user_selected_weight").checked = saved_selection.user_selected;
      by_id("chart_weight_model_feature").value = String(saved_selection.model_feature);
      by_id("chart_weight_intermediate_feature").value = String(saved_selection.intermediate_feature);
      update_matched_weight_controls();
    };

    const selection_validation_error = selection => {
      if (selection.model_feature < 0 || selection.intermediate_feature < 0) {
        return "Weight feature indices must be non-negative integers.";
      }
      if (!selection.user_selected) return null;
      const chart_name = app.axis_chart_name;
      const capability = matched_weight_capability(chart_name);
      if (!capability.available) return capability.reason;
      if (
        selection.model_feature > capability.maximum
        || selection.intermediate_feature > capability.maximum
      ) {
        return `Both weight feature indices must be between 0 and ${capability.maximum}.`;
      }
      return null;
    };

    const save_selection = async selection => {
      const run_id = String(app.current_run_id || "");
      if (!run_id) throw new Error("Select a run before changing the weight coupling");
      const response = await fetch(
        `/api/weight-selection?run=${encodeURIComponent(run_id)}`,
        {
        method: "POST",
        cache: "no-store",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(selection),
        },
      );
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || `${response.status} ${response.statusText}`);
      if (run_id !== String(app.current_run_id || "")) {
        throw new Error("The selected run changed while saving the weight coupling");
      }
      saved_selection = normalise_selection(value);
      selection_loaded = true;
      loaded_run_id = run_id;
      return saved_selection;
    };

    const load_selection = async () => {
      const run_id = String(app.current_run_id || "");
      const request_serial = ++selection_request_serial;
      if (!run_id) {
        saved_selection = {...default_selection};
        selection_loaded = false;
        loaded_run_id = "";
        write_selection_controls();
        return;
      }
      try {
        const response = await fetch(
          `/api/weight-selection?run=${encodeURIComponent(run_id)}`,
          {cache: "no-store"},
        );
        const value = await response.json();
        if (!response.ok) throw new Error(value.error || `${response.status} ${response.statusText}`);
        if (
          request_serial !== selection_request_serial
          || run_id !== String(app.current_run_id || "")
        ) return;
        saved_selection = normalise_selection(value);
        selection_loaded = true;
        loaded_run_id = run_id;
        write_selection_controls();
        if (app.figures) {
          render_figures().catch(error => show_toast(`Matched weight refresh failed: ${error.message}`));
        }
      } catch (error) {
        if (request_serial !== selection_request_serial) return;
        selection_loaded = false;
        loaded_run_id = "";
        update_matched_weight_controls();
        show_toast(`Matched weight settings unavailable: ${error.message}`);
      }
    };

    const weight_trace_kind = trace => {
      const meta = trace?.meta;
      if (!meta || typeof meta !== "object" || Array.isArray(meta)) return null;
      if (meta.instra_weight_selection_protocol !== protocol) return null;
      return String(meta.instra_weight_selection_kind || "random");
    };

    const base_prepare_figure_matched_weights = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_matched_weights(figure, chart_name);
      if (!weight_chart_set.has(chart_name)) return prepared;
      const current_only = resolved_current_only(chart_name);
      const capability = matched_weight_capability(chart_name);
      const viewer_selection = window.__instra_weight_viewer_selection?.selection?.()
        || saved_selection;
      const selected = viewer_selection.user_selected === true;

      prepared.data = (prepared.data || []).filter(trace => {
        const kind = weight_trace_kind(trace);
        if (kind === null) return true;
        if (!current_only) return kind !== "user";
        if (!selected || !capability.available) return kind !== "user";
        const meta = trace.meta || {};
        return (
          (kind === "user" || kind === "user_random")
          && finite_integer(meta.instra_weight_model_feature) === finite_integer(viewer_selection.model_feature)
          && finite_integer(meta.instra_weight_intermediate_feature) === finite_integer(viewer_selection.intermediate_feature)
        );
      });
      return prepared;
    };

    install_fields();

    const base_populate_chart_settings_form_matched = populate_chart_settings_form;
    populate_chart_settings_form = function(chart_name, supplied = null) {
      const result = base_populate_chart_settings_form_matched(chart_name, supplied);
      if (weight_chart_set.has(chart_name)) write_selection_controls();
      else if (by_id("chart_matched_weight_section")) by_id("chart_matched_weight_section").hidden = true;
      return result;
    };

    const base_sync_chart_setting_outputs_matched = sync_chart_setting_outputs;
    sync_chart_setting_outputs = function() {
      const result = base_sync_chart_setting_outputs_matched();
      update_matched_weight_controls();
      return result;
    };

    document.addEventListener("click", event => {
      const button = event.target.closest("#save_chart_settings");
      if (!button || save_bypass || save_in_flight) return;
      const chart_name = app.axis_chart_name;
      if (!weight_chart_set.has(chart_name) || !selection_loaded) return;

      const draft = selection_from_controls();
      if (same_selection(draft, saved_selection)) return;

      const error = selection_validation_error(draft);
      if (error) {
        event.preventDefault();
        event.stopImmediatePropagation();
        by_id("chart_settings_error").textContent = error;
        by_id("chart_settings_error").hidden = false;
        update_matched_weight_controls();
        return;
      }

      event.preventDefault();
      event.stopImmediatePropagation();
      save_in_flight = true;
      button.disabled = true;
      save_selection(draft)
        .then(() => {
          save_bypass = true;
          button.disabled = false;
          button.click();
          save_bypass = false;
          if (draft.user_selected) {
            show_toast(
              `Matched weight set to model ${draft.model_feature}, intermediate ${draft.intermediate_feature}; `
              + "it will appear at the next weight snapshot."
            );
          } else {
            show_toast("Matched weight returned to random selection.");
          }
        })
        .catch(error => {
          button.disabled = false;
          by_id("chart_settings_error").textContent = `Matched weight save failed: ${error.message}`;
          by_id("chart_settings_error").hidden = false;
        })
        .finally(() => {
          save_in_flight = false;
          update_matched_weight_controls();
        });
    }, true);

    const style = document.createElement("style");
    style.textContent = `
      .matched-weight-section {
        margin: 4px 0 10px;
        padding: 10px 12px;
        border: 1px solid #d7dde4;
        border-radius: 6px;
        background: #fbfcfd;
      }
      .matched-weight-section.unavailable { background: #f5f6f7; }
      .matched-weight-coordinate-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 8px;
        margin-top: 8px;
      }
      .matched-weight-coordinate-grid input { width: 100%; box-sizing: border-box; }
      .matched-weight-range, .matched-weight-note {
        margin: 7px 0 0;
        font-size: 11px;
        line-height: 1.35;
        color: #697480;
      }
      .matched-weight-section.invalid .matched-weight-range {
        color: #b42318;
        font-weight: 600;
      }
      .matched-weight-section.unavailable .matched-weight-coordinate-grid,
      .matched-weight-section.unavailable .matched-weight-note {
        opacity: .62;
      }
    `;
    document.head.appendChild(style);

    window.__instra_matched_weight_selection = {
      capability: matched_weight_capability,
      selection: () => ({...saved_selection}),
      save: (model_feature, intermediate_feature) => save_selection(normalise_selection({
        user_selected: true,
        model_feature,
        intermediate_feature,
      })),
      reload: load_selection,
      run_id: () => loaded_run_id,
    };

    const base_select_run_matched_selection = select_run;
    select_run = function(run_id, options = {}) {
      const changing = String(run_id || "") !== String(app.current_run_id || "");
      const result = base_select_run_matched_selection(run_id, options);
      if (changing || loaded_run_id !== String(app.current_run_id || "")) {
        saved_selection = {...default_selection};
        selection_loaded = false;
        loaded_run_id = "";
        write_selection_controls();
        queueMicrotask(load_selection);
      }
      return result;
    };

    setInterval(() => {
      if (!by_id("chart_settings_overlay")?.hidden && weight_chart_set.has(app.axis_chart_name)) {
        update_matched_weight_controls();
      }
    }, 500);

    load_selection();
  }, 0);
});
// ^^^ THOG
