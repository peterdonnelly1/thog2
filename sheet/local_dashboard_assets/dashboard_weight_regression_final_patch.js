// vvv THOG
"use strict";

// Final narrow regression guard for the stacked Heatmap/Weights presentation,
// editable step-window fields, retained display ranges, trajectory headroom,
// run-table width allocation, and matched-coordinate RND history.
window.addEventListener("load", () => {
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
    const protocol = "matched_six_v1";

    const finite_integer = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isInteger(numeric) ? numeric : null;
    };

    const current_step_context = () => {
      const stability = window.__instra_weight_stability_final;
      if (typeof stability?.context_key === "function") {
        const key = String(stability.context_key() || "");
        if (key) return key;
      }
      if (app.workspace_mode === true) return "workspace:pending";
      return app.current_run_id ? `run:${String(app.current_run_id)}` : "";
    };

    // A step input owns its typed draft until it is explicitly committed/cleared or
    // the selected run/workspace context changes.  Blur is not a commit operation.
    const protected_step_inputs = new WeakMap();
    const native_value = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
    const protect_step_input = input => {
      if (!input || protected_step_inputs.has(input) || !native_value?.get || !native_value?.set) return;
      const state = {dirty: false, context: current_step_context()};
      Object.defineProperty(input, "value", {
        configurable: true,
        enumerable: true,
        get() { return native_value.get.call(input); },
        set(value) {
          const context = current_step_context();
          if (state.context !== context) {
            state.context = context;
            state.dirty = false;
          }
          if (state.dirty) return;
          native_value.set.call(input, value);
        },
      });
      input.addEventListener("input", () => {
        state.context = current_step_context();
        state.dirty = true;
      });
      protected_step_inputs.set(input, state);
    };
    const protect_step_inputs = () => {
      protect_step_input(by_id("weight_step_from"));
      protect_step_input(by_id("weight_step_to"));
    };
    const clear_step_input_drafts = () => {
      const context = current_step_context();
      for (const id of ["weight_step_from", "weight_step_to"]) {
        const state = protected_step_inputs.get(by_id(id));
        if (!state) continue;
        state.context = context;
        state.dirty = false;
      }
    };
    window.__instra_clear_weight_step_input_drafts = clear_step_input_drafts;
    const reconcile_step_input_drafts = () => {
      const stability = window.__instra_weight_stability_final;
      const range = typeof stability?.selected_range === "function" ? stability.selected_range() : null;
      const from = by_id("weight_step_from");
      const to = by_id("weight_step_to");
      const from_state = protected_step_inputs.get(from);
      const to_state = protected_step_inputs.get(to);
      const context = current_step_context();
      for (const state of [from_state, to_state]) {
        if (state && state.context !== context) {
          state.context = context;
          state.dirty = false;
        }
      }
      if (!range || !from || !to) return;
      if (
        finite_integer(from.value) === finite_integer(range.minimum)
        && finite_integer(to.value) === finite_integer(range.maximum)
      ) {
        clear_step_input_drafts();
      }
    };
    protect_step_inputs();

    const prepare_step_button_command = target => {
      const apply = target?.closest?.("#weight_step_apply");
      const whole = target?.closest?.("#weight_step_whole_range");
      const latest = target?.closest?.("#weight_step_latest");
      if (!apply && !whole && !latest) return false;
      if (whole || latest) {
        clear_step_input_drafts();
        return true;
      }
      const minimum = finite_integer(by_id("weight_step_from")?.value);
      const maximum = finite_integer(by_id("weight_step_to")?.value);
      if (minimum !== null && maximum !== null && minimum >= 0 && maximum >= minimum) {
        clear_step_input_drafts();
      }
      return true;
    };

    // Show/Enter commits the draft; Whole range explicitly clears it. Pointerdown
    // runs before the older capture-phase click owner calls sync_header(). Keyboard
    // activation is handled separately because it does not produce pointerdown.
    window.addEventListener("pointerdown", event => {
      prepare_step_button_command(event.target);
    }, true);
    window.addEventListener("keydown", event => {
      if (!["Enter", " ", "Spacebar"].includes(event.key)) return;
      prepare_step_button_command(event.target);
    }, true);
    window.addEventListener("keyup", event => {
      if (event.key !== "Enter" || !event.target.matches?.("#weight_step_from, #weight_step_to")) return;
      reconcile_step_input_drafts();
    }, true);

    // Normalize only the render copy: selection_kind records how the coordinate was
    // obtained, not which historical snapshots belong to the selected coordinate.
    // The matched-weight layer can therefore select by coordinate while the stable
    // owner independently applies current-only / explicit-range / history semantics.
    const route_selected_weight_coordinate = (figure, chart_name) => {
      if (!weight_chart_set.has(chart_name) || !figure || typeof figure !== "object") return figure;
      const selection_api = window.__instra_weight_viewer_selection
        || window.__instra_matched_weight_selection;
      const selection = typeof selection_api?.selection === "function" ? selection_api.selection() : null;
      if (!selection || selection.user_selected !== true) return figure;
      const model_feature = finite_integer(selection.model_feature);
      const intermediate_feature = finite_integer(selection.intermediate_feature);
      if (model_feature === null || intermediate_feature === null) return figure;
      let routed;
      try {
        routed = typeof clone_figure === "function"
          ? clone_figure(figure)
          : JSON.parse(JSON.stringify(figure));
      } catch (_error) {
        return figure;
      }
      for (const trace of routed.data || []) {
        const meta = trace?.meta;
        if (!meta || typeof meta !== "object" || Array.isArray(meta)) continue;
        if (meta.instra_weight_selection_protocol !== protocol) continue;
        if (
          finite_integer(meta.instra_weight_model_feature) !== model_feature
          || finite_integer(meta.instra_weight_intermediate_feature) !== intermediate_feature
        ) continue;
        trace.meta = {...meta, instra_weight_selection_kind: "user_random"};
      }
      return routed;
    };

    // Render-only geometry repairs. Stored axis settings remain unchanged.
    const padded_figures = new WeakSet();
    const base_prepare_figure_regression = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const routed = route_selected_weight_coordinate(figure, chart_name);
      const prepared = base_prepare_figure_regression(routed, chart_name);
      if (!prepared || typeof prepared !== "object") return prepared;

      if (chart_name === "heatmap") {
        const heatmap_trace = (prepared.data || []).find(trace => trace?.type === "heatmap");
        if (heatmap_trace) {
          prepared.layout = {...(prepared.layout || {})};
          prepared.layout.margin = {
            ...(prepared.layout.margin || {}),
            r: Math.max(220, Number(prepared.layout?.margin?.r || 0)),
          };
          const shell = document.querySelector('.chart-card[data-chart="heatmap"] .heatmap-shell');
          const shell_width = Math.max(0, Number(shell?.clientWidth || 0));
          const plot_width = Math.max(
            1,
            shell_width
              - Number(prepared.layout.margin.l || 0)
              - Number(prepared.layout.margin.r || 0),
          );
          const key_offset_px = 48;
          heatmap_trace.colorbar = {
            ...(heatmap_trace.colorbar || {}),
            x: shell_width > 0 ? 1 + key_offset_px / plot_width : 1.08,
            xanchor: "left",
            xpad: 8,
          };
        }
        return prepared;
      }

      if (!weight_chart_set.has(chart_name) || padded_figures.has(prepared)) return prepared;
      const range = prepared.layout?.yaxis?.range;
      if (
        Array.isArray(range)
        && range.length === 2
        && Number.isFinite(Number(range[0]))
        && Number.isFinite(Number(range[1]))
        && Number(range[1]) > Number(range[0])
      ) {
        const lower = Number(range[0]);
        const upper = Number(range[1]);
        const span = upper - lower;
        prepared.layout = {...prepared.layout};
        prepared.layout.yaxis = {
          ...prepared.layout.yaxis,
          range: [lower, upper + Math.max(span * 0.025, Math.abs(upper) * 1e-9, 1e-12)],
        };
      }
      padded_figures.add(prepared);
      return prepared;
    };

    // RND chooses an already-retained random coupling, not a coordinate that may
    // only become available in a later snapshot. Manual +/- controls retain their
    // existing request-next-snapshot semantics.
    let random_save_in_flight = false;
    const current_selection = () => {
      const api = window.__instra_matched_weight_selection;
      const value = typeof api?.selection === "function" ? api.selection() : null;
      return value && typeof value === "object"
        ? value
        : {user_selected: false, model_feature: 0, intermediate_feature: 0};
    };
    const retained_weight_pairs = () => {
      const pairs = new Map();
      const pairs_by_run = new Map();
      const figures = [];
      for (const chart_name of weight_chart_names) {
        try {
          const figure = figure_for_chart(chart_name);
          if (figure) figures.push(figure);
        } catch (_error) {}
      }
      for (const mount of document.querySelectorAll(".js-plotly-plot")) {
        if (Array.isArray(mount.data)) figures.push({data: mount.data});
      }
      for (const figure of figures) {
        for (const trace of figure?.data || []) {
          const meta = trace?.meta;
          if (!meta || typeof meta !== "object" || Array.isArray(meta)) continue;
          if (meta.instra_weight_selection_protocol !== protocol) continue;
          if (String(meta.instra_weight_selection_kind || "random") === "user") continue;
          const model_feature = finite_integer(meta.instra_weight_model_feature);
          const intermediate_feature = finite_integer(meta.instra_weight_intermediate_feature);
          if (model_feature === null || intermediate_feature === null) continue;
          if (model_feature < 0 || intermediate_feature < 0) continue;
          const key = `${model_feature}:${intermediate_feature}`;
          const pair = {model_feature, intermediate_feature};
          pairs.set(key, pair);
          const run_id = meta.instra_workspace_run_id ? String(meta.instra_workspace_run_id) : null;
          if (run_id) {
            if (!pairs_by_run.has(run_id)) pairs_by_run.set(run_id, new Set());
            pairs_by_run.get(run_id).add(key);
          }
        }
      }
      if (app.workspace_mode === true && pairs_by_run.size > 1) {
        const run_sets = [...pairs_by_run.values()];
        const common = [...run_sets[0]].filter(key => run_sets.every(run_pairs => run_pairs.has(key)));
        return common.map(key => pairs.get(key)).filter(Boolean);
      }
      return [...pairs.values()];
    };
    const same_pair = (pair, selection) => (
      pair.model_feature === finite_integer(selection?.model_feature)
      && pair.intermediate_feature === finite_integer(selection?.intermediate_feature)
    );
    const wait_for_selection = async (expected, timeout_ms = 5000) => {
      const deadline = Date.now() + timeout_ms;
      while (Date.now() < deadline) {
        const current = current_selection();
        if (
          current.user_selected === true
          && finite_integer(current.model_feature) === expected.model_feature
          && finite_integer(current.intermediate_feature) === expected.intermediate_feature
        ) return true;
        await new Promise(resolve => setTimeout(resolve, 25));
      }
      return false;
    };
    const persist_retained_pair = async pair => {
      if (random_save_in_flight) return false;
      if (!by_id("chart_settings_overlay")?.hidden) {
        show_toast("Close chart settings before choosing a random retained weight.");
        return false;
      }
      const save_button = by_id("save_chart_settings");
      const user_toggle = by_id("chart_user_selected_weight");
      const model_input = by_id("chart_weight_model_feature");
      const intermediate_input = by_id("chart_weight_intermediate_feature");
      if (!save_button || !user_toggle || !model_input || !intermediate_input) {
        show_toast("Matched-weight controls are not ready yet.");
        return false;
      }

      random_save_in_flight = true;
      const previous_axis_chart_name = app.axis_chart_name;
      const previous_axis_workspace_mode = app.axis_chart_workspace_mode;
      const previous_show_toast = show_toast;
      let first_click_seen = false;
      const stop_completion_click = event => {
        const button = event.target.closest?.("#save_chart_settings");
        if (!button) return;
        if (!first_click_seen) {
          first_click_seen = true;
          return;
        }
        event.preventDefault();
        event.stopImmediatePropagation();
      };
      show_toast = message => {
        const text = String(message || "");
        if (text.startsWith("Matched weight set to model ")) return;
        previous_show_toast(message);
      };
      window.addEventListener("click", stop_completion_click, true);
      try {
        app.axis_chart_name = weight_chart_names.find(chart_name => {
          try { return Boolean(figure_for_chart(chart_name)); }
          catch (_error) { return false; }
        }) || weight_chart_names[0];
        app.axis_chart_workspace_mode = app.workspace_mode === true;
        user_toggle.checked = true;
        model_input.value = String(pair.model_feature);
        intermediate_input.value = String(pair.intermediate_feature);
        save_button.disabled = false;
        save_button.click();
        const saved = await wait_for_selection(pair);
        await new Promise(resolve => setTimeout(resolve, 60));
        if (!saved) throw new Error("matched-weight save did not complete");
        previous_show_toast(
          `Random retained weight: residual ${pair.model_feature}, branch ${pair.intermediate_feature}.`
        );
        return true;
      } catch (error) {
        previous_show_toast(`Random retained-weight selection failed: ${error.message}`);
        return false;
      } finally {
        window.removeEventListener("click", stop_completion_click, true);
        show_toast = previous_show_toast;
        app.axis_chart_name = previous_axis_chart_name;
        app.axis_chart_workspace_mode = previous_axis_workspace_mode;
        random_save_in_flight = false;
      }
    };

    window.addEventListener("click", event => {
      const button = event.target.closest?.("#weight_random_jump");
      if (!button) return;
      // The run-aware viewer installed by the range-interaction owner supersedes
      // this retained-capture implementation.  Yield without stopping propagation
      // so its later capture listener can choose from this run's matrix dimensions
      // (or from recorded pairs for a completed historical run).
      if (window.__instra_weight_range_interaction_final) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      const retained = retained_weight_pairs();
      if (!retained.length) {
        show_toast("No retained weight couplings are available yet.");
        return;
      }
      const selection = current_selection();
      const alternatives = retained.filter(pair => !same_pair(pair, selection));
      const choices = alternatives.length ? alternatives : retained;
      const pair = choices[Math.floor(Math.random() * choices.length)];
      persist_retained_pair(pair);
    }, true);

    const classify_run_table_headers = () => {
      const header_row = document.querySelector(".runs-table thead tr");
      if (!header_row) return;
      const classes = new Map([
        ["W&B ID", "instra-wandb-column"],
        ["State", "instra-state-column"],
        ["Host", "instra-host-column"],
        ["t", "instra-duration-column"],
        ["Updated", "instra-updated-column"],
      ]);
      for (const header of header_row.children) {
        const class_name = classes.get(String(header.textContent || "").trim());
        if (class_name) header.classList.add(class_name);
      }
    };

    // The old single-group 100% minimum height is wrong for the stacked synthetic
    // groups. Expanded groups size to their contents; collapsed/maximized states
    // retain their established rules. In the run table NAME is deliberately the
    // only elastic descriptive column; bounded fields do not consume widened panes.
    const style = document.createElement("style");
    style.id = "thog2_weight_regression_final_style";
    style.textContent = `
      #heatmap_chart_group:not(.collapsed):not(.maximized),
      #coefficients_chart_group:not(.collapsed):not(.maximized) {
        min-height: 0 !important;
      }
      #heatmap_chart_group:not(.collapsed):not(.maximized) > .chart-grid,
      #coefficients_chart_group:not(.collapsed):not(.maximized) > .chart-grid {
        min-height: 0 !important;
      }
      .runs-table .name-column { width: auto !important; }
      .runs-table .instra-wandb-column { width: 92px !important; }
      .runs-table .instra-state-column { width: 88px !important; }
      .runs-table .instra-host-column { width: 96px !important; }
      .runs-table .duration-column,
      .runs-table .instra-duration-column {
        width: 78px !important;
        min-width: 78px !important;
        text-transform: none !important;
        white-space: nowrap;
      }
      .runs-table .instra-updated-column { width: 110px !important; }
      .run-overview-pane .overview-metadata {
        grid-template-columns: 120px minmax(0, 1fr) !important;
      }
      .run-overview-pane .overview-hardware-grid {
        grid-template-columns: 104px minmax(0, 1fr) !important;
      }
      .run-overview-pane .overview-key-row {
        grid-template-columns: minmax(155px, .65fr) minmax(180px, 1.35fr) !important;
      }
    `;
    document.head.appendChild(style);
    classify_run_table_headers();

    // New installations/default-only profiles move from 11px to 12px. Any
    // explicitly non-default current choice survives unchanged.
    const overview_default_key = "thog2_local_overview_default_font_size";
    const overview_current_key = "thog2_local_overview_font_size";
    const stored_default = localStorage.getItem(overview_default_key);
    const stored_current = localStorage.getItem(overview_current_key);
    if (stored_default === null) {
      localStorage.setItem(overview_default_key, "12");
      if (stored_current === null || Number(stored_current) === 11) {
        const larger = by_id("overview_font_larger");
        if (larger) larger.click();
        else {
          localStorage.setItem(overview_current_key, "12");
          const pane = by_id("run_overview_pane") || document.querySelector(".run-overview-pane");
          pane?.style.setProperty("--thog2-overview-font-size", "12px");
        }
      }
    }

    const observer = new MutationObserver(() => {
      protect_step_inputs();
      classify_run_table_headers();
    });
    observer.observe(document.body, {childList: true, subtree: true});

    // Status and the dependency-gated stable owner can arrive after this last-loaded
    // asset. Bounded lightweight reconciliation protects typed range drafts while
    // the run list and controls settle.
    let reconciliation_passes = 0;
    const reconciliation_timer = setInterval(() => {
      reconciliation_passes += 1;
      protect_step_inputs();
      reconcile_step_input_drafts();
      classify_run_table_headers();
      if (reconciliation_passes >= 240) clearInterval(reconciliation_timer);
    }, 250);
  }, 360);
});
// ^^^ THOG
