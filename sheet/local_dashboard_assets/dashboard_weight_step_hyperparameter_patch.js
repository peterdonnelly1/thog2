// vvv THOG
"use strict";

// Seed the existing Weights step-window controller once per displayed context from
// run hyperparameters.  The existing group header remains authoritative after that
// seed: user edits, including "whole range", are never reapplied or overwritten.
window.addEventListener("load", () => {
  setTimeout(() => {
    const start_key = "instrumentation__depth_weight_curves__start_step";
    const end_key = "instrumentation__depth_weight_curves__end_step";
    const attempted_contexts = new Set();

    const finite_step = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isInteger(numeric) && numeric >= 0 ? numeric : null;
    };

    const run_id = run => String(
      run?.dashboard_run_id
      || run?.local_run_id
      || run?.wandb_run_id
      || run?.artifact_name
      || run?.run_name
      || "",
    );

    const fresh_run = run => {
      if (!run) return null;
      const identifier = run_id(run);
      if (
        typeof app !== "undefined"
        && identifier
        && identifier === String(app.current_run_id || "")
        && app.current_status
        && typeof app.current_status === "object"
      ) {
        return {
          ...run,
          ...app.current_status,
          configuration: run.configuration || app.current_status.configuration || {},
        };
      }
      return run;
    };

    const context_runs = () => {
      if (typeof app === "undefined") return [];
      if (app.workspace_mode === true && typeof window.__instra_workspace?.visible_runs === "function") {
        return window.__instra_workspace.visible_runs().map(fresh_run).filter(Boolean);
      }
      const run = typeof current_run === "function" ? fresh_run(current_run()) : null;
      return run ? [run] : [];
    };

    const context_key = runs => {
      if (!runs.length || typeof app === "undefined") return "";
      if (app.workspace_mode === true) {
        return `workspace:${runs.map(run_id).sort().join("|")}`;
      }
      return `run:${run_id(runs[0])}`;
    };

    const configured_pair = run => {
      const configuration = run?.configuration && typeof run.configuration === "object"
        ? run.configuration
        : {};
      const has_start = Object.prototype.hasOwnProperty.call(configuration, start_key);
      const has_end = Object.prototype.hasOwnProperty.call(configuration, end_key);
      if (!has_start && !has_end) return {known: false, range: null};
      const minimum = finite_step(configuration[start_key]);
      const maximum = finite_step(configuration[end_key]);
      if (minimum === null && maximum === null) return {known: true, range: null};
      if (minimum === null || maximum === null || maximum < minimum) {
        return {known: true, invalid: true, range: null};
      }
      return {known: true, invalid: false, range: {minimum, maximum}};
    };

    const common_configured_range = runs => {
      const pairs = runs.map(configured_pair);
      if (!pairs.length || pairs.some(pair => !pair.known)) return {ready: false, range: null};
      if (pairs.some(pair => pair.invalid)) return {ready: true, invalid: true, range: null};
      const ranges = pairs.map(pair => pair.range);
      if (ranges.every(range => range === null)) return {ready: true, invalid: false, range: null};
      if (ranges.some(range => range === null)) return {ready: true, invalid: true, range: null};
      const first = ranges[0];
      const identical = ranges.every(range => (
        range.minimum === first.minimum && range.maximum === first.maximum
      ));
      return identical
        ? {ready: true, invalid: false, range: first}
        : {ready: true, invalid: true, range: null};
    };

    const maybe_seed_step_range = () => {
      const api = window.__instra_weight_controls_v2;
      if (!api || typeof api.set_step_range !== "function") return;
      const runs = context_runs();
      const key = context_key(runs);
      if (!key || attempted_contexts.has(key)) return;

      // Any already-active range is a user/existing-controller decision and wins.
      if (typeof api.selected_step_range === "function" && api.selected_step_range()) {
        attempted_contexts.add(key);
        return;
      }

      const configured = common_configured_range(runs);
      if (!configured.ready) return;
      attempted_contexts.add(key);
      if (!configured.range) {
        if (configured.invalid) console.warn("INSTRA ignored inconsistent or incomplete configured Weights step range.");
        return;
      }

      const capacity = Number(api.common_history_capacity?.());
      const width = configured.range.maximum - configured.range.minimum + 1;
      if (Number.isFinite(capacity) && capacity > 0 && width > capacity) {
        console.warn(`INSTRA ignored configured Weights step range ${configured.range.minimum}-${configured.range.maximum}: ${width} steps exceeds retained history ${capacity}.`);
        return;
      }
      api.set_step_range(configured.range.minimum, configured.range.maximum);
    };

    const format_current_step_label = () => {
      const current = document.getElementById("weight_step_current");
      if (!current) return;
      const text = String(current.textContent || "").trim();
      if (!text || (text.startsWith("(") && text.endsWith(")"))) return;
      if (text.startsWith("current step")) current.textContent = `(${text})`;
    };

    const style = document.createElement("style");
    style.textContent = `
      .weight-step-current { font-weight: 400 !important; }
      #heatmap_chart_group:not(.collapsed):not(.maximized),
      #coefficients_chart_group:not(.collapsed):not(.maximized) {
        min-height: 0 !important;
      }
      #heatmap_chart_group:not(.collapsed):not(.maximized) > .chart-grid,
      #coefficients_chart_group:not(.collapsed):not(.maximized) > .chart-grid {
        min-height: 0 !important;
      }
    `;
    document.head.appendChild(style);

    const observer = new MutationObserver(() => {
      format_current_step_label();
      maybe_seed_step_range();
    });
    observer.observe(document.body, {subtree: true, childList: true, characterData: true});

    format_current_step_label();
    maybe_seed_step_range();
    window.setInterval(maybe_seed_step_range, 500);
  }, 0);
});
// ^^^ THOG
