// vvv THOG
"use strict";

(function install_processing_sep16_stability_polish() {
  const style_id = "instra-processing-sep16-stability-polish-style";
  if (!document.getElementById(style_id)) {
    const style = document.createElement("style");
    style.id = style_id;
    style.textContent = `
      .runs-table .eye-button.instra-processing-paired {
        color:#1378d4 !important;
        filter:none !important;
      }
      .runs-table .eye-button.instra-processing-paired.off {
        color:#1378d4 !important;
        opacity:1 !important;
      }
      .processing-throughput-card .chart-card-header { position:relative !important; }
      .processing-throughput-card .processing-throughput-z-button {
        position:absolute !important;
        left:50% !important;
        top:50% !important;
        transform:translate(-50%, -50%) !important;
        margin:0 !important;
        font-weight:400 !important;
      }
    `;
    document.head.appendChild(style);
  }

  app.processing_paired_run_ids = app.processing_paired_run_ids || new Set();
  app.processing_pair_roles = app.processing_pair_roles || {};

  function same_set(left, right) {
    if (left.size !== right.size) return false;
    for (const value of left) if (!right.has(value)) return false;
    return true;
  }

  function decorate_paired_eyes() {
    const paired = app.processing_paired_run_ids || new Set();
    for (const row of document.querySelectorAll('.runs-table tbody tr[data-run-id]')) {
      const run_id = String(row.dataset.runId || "");
      const eye = row.querySelector('.visibility-column .eye-button');
      if (!eye) continue;
      const is_paired = paired.has(run_id);
      eye.classList.toggle("instra-processing-paired", is_paired);
      if (!is_paired) continue;
      const role = app.processing_pair_roles?.[run_id] || "paired profiler run";
      eye.title = role;
      eye.setAttribute("aria-label", role);
    }
  }

  const render_runs_before_processing_pair_decoration = render_runs;
  render_runs = function() {
    const result = render_runs_before_processing_pair_decoration();
    decorate_paired_eyes();
    return result;
  };

  function apply_processing_pair(payload, trace_available) {
    const source = payload?.premat_compatibility_source;
    const companion_id = String(source?.dashboard_run_id || "");
    const selected_id = String(processing_current_run() || "");
    const next = new Set();
    const roles = {};
    if (trace_available && selected_id && companion_id) {
      next.add(selected_id);
      next.add(companion_id);
      roles[selected_id] = "NSYS source · paired Processing evidence";
      roles[companion_id] = "NCU companion · automatically paired with selected NSYS run";
    }

    let visibility_changed = false;
    for (const run_id of next) {
      if (!is_visible(run_id)) {
        app.visibility[run_id] = true;
        visibility_changed = true;
      }
    }
    if (visibility_changed) save_json("thog2_local_run_visibility", app.visibility);

    const set_changed = !same_set(app.processing_paired_run_ids || new Set(), next);
    app.processing_paired_run_ids = next;
    app.processing_pair_roles = roles;
    if (visibility_changed || set_changed) render_runs();
    else decorate_paired_eyes();
  }

  // Exact attribution panes. Device-wide NSYS counters cannot be numerically
  // split when both streams are simultaneously active, nor when one sampling
  // bin contains mixed/unknown activity. Do not relabel those cases as if they
  // were stream-specific measurements.
  processing_resource_panes = function(rows) {
    const definitions = [
      {key:"main", label:"MAIN Stream", states:new Set(["MAIN_ONLY"])},
      {key:"premat", label:"PREMAT Stream", states:new Set(["PREMAT_ONLY"])},
      {key:"overlap", label:"MAIN/PREMAT Overlap", states:new Set(["MAIN_PREMAT_OVERLAP"])},
      {key:"ambiguous", label:"Mixed / Unattributed", states:new Set(["MIXED_SEQUENTIAL", "OTHER_OR_UNKNOWN"])},
    ];
    const active = definitions.filter(definition => rows.some(row => definition.states.has(String(row.attribution_state))));
    if (!active.length) return [];
    const gap = 0.055;
    const usable = 1 - gap * Math.max(0, active.length - 1);
    const height = usable / active.length;
    return active.map((definition, index) => {
      const top = 1 - index * (height + gap);
      const bottom = Math.max(0, top - height);
      return {
        ...definition,
        domain:[bottom, top],
        yaxis:index === 0 ? "y" : `y${index + 1}`,
      };
    });
  };

  function polish_processing_labels() {
    const resource = by_id("processing_resource_card");
    const resource_heading = resource?.querySelector(".chart-heading-copy h2");
    if (resource_heading) resource_heading.textContent = "Stream Resource Contention";
    const resource_copy = resource?.querySelector(".chart-heading-copy > p:not(#processing_resource_idle_summary)");
    if (resource_copy) {
      resource_copy.textContent = "Horizontal axis: capture time (ms). MAIN/PREMAT Overlap means both streams had kernels simultaneously active inside that NSYS sample bin; device-wide counters are therefore shown combined, not assigned to either stream.";
    }

    const timeline = by_id("processing_timeline_card");
    const timeline_heading = timeline?.querySelector(".chart-heading-copy h2");
    if (timeline_heading) timeline_heading.textContent = "MAIN / PREMAT Operations";

    const compatibility = by_id("processing_compatibility_card");
    const compatibility_heading = compatibility?.querySelector(".chart-heading-copy h2");
    if (compatibility_heading) compatibility_heading.textContent = "PREMAT Compatibility";
    const compatibility_copy = compatibility?.querySelector(".chart-heading-copy > p:not(.processing-compatibility-source)");
    if (compatibility_copy) {
      compatibility_copy.textContent = "NCU structural test: can the measured MAIN and PREMAT blocks coexist on one SM under register, shared-memory, warp, thread and block-slot limits? Colour applies only to the profiled MAIN operation/layer; blank means unmeasured. This is scheduling capability, not observed overlap.";
    }

    if (typeof chart_titles === "object") {
      chart_titles.processing_resource = "Stream Resource Contention";
      chart_titles.processing_timeline = "MAIN / PREMAT Operations";
      chart_titles.processing_compatibility = "PREMAT Compatibility";
    }
  }

  // Keep synchronized zoom/pan, but deliberately do not propagate Plotly hover
  // events between charts. Plotly.Fx.hover can generate reciprocal hover events
  // and was a plausible long-session feedback/leak source.
  function extract_xrange(event) {
    if (event?.["xaxis.autorange"] === true) return {"xaxis.autorange": true};
    const left = Number(event?.["xaxis.range[0]"]);
    const right = Number(event?.["xaxis.range[1]"]);
    if (!Number.isFinite(left) || !Number.isFinite(right)) return null;
    return {"xaxis.range":[left, right]};
  }

  function synchronized_mounts() {
    return [
      by_id("processing_resource_plot"),
      by_id("processing_resource_clock_plot"),
      by_id("processing_timeline_plot"),
      by_id("processing_resource_compatibility_plot"),
    ].filter(mount => mount && mount.dataset.plotReady === "true" && mount.offsetParent !== null);
  }

  processing_gpu_link_time_axes = function() {
    for (const source of synchronized_mounts()) {
      if (source._processing_stable_axis_linked === true) continue;
      source._processing_stable_axis_linked = true;
      source.on("plotly_relayout", event => {
        if (processing_view.gpu_axis_sync) return;
        const update = extract_xrange(event || {});
        if (!update) return;
        processing_view.gpu_axis_sync = true;
        Promise.all(
          synchronized_mounts()
            .filter(target => target !== source)
            .map(target => Promise.resolve(Plotly.relayout(target, update)))
        ).finally(() => { processing_view.gpu_axis_sync = false; });
      });
    }
  };
  processing_resource_link_time_axes = processing_gpu_link_time_axes;

  // Finished-run timing plots are immutable. The legacy 2-second timer remains
  // installed in dashboard_processing.js, so make repeated calls no-ops unless
  // the relevant run cohort/revision actually changed. Also serialize renders.
  const processing_render_update_timing_before_stability = processing_render_update_timing;
  let timing_fingerprint = null;
  let timing_render_in_flight = false;
  let timing_render_queued = false;

  function current_timing_fingerprint() {
    const runs = typeof processing_update_timing_ordered_runs === "function"
      ? processing_update_timing_ordered_runs()
      : [];
    return JSON.stringify({
      workspace:Boolean(app.workspace_mode),
      current:String(app.current_run_id || ""),
      runs:runs.map(run => ({
        id:String(run_identifier(run)),
        revision:run.revision || null,
        state:String(run.run_state || ""),
        visible:is_visible(run_identifier(run)),
      })),
    });
  }

  processing_render_update_timing = async function(force = false) {
    const fingerprint = current_timing_fingerprint();
    const runs = typeof processing_update_timing_ordered_runs === "function"
      ? processing_update_timing_ordered_runs()
      : [];
    const any_active = runs.some(run => is_active_run_state(run.run_state));
    if (!force && !any_active && fingerprint === timing_fingerprint) return;
    if (timing_render_in_flight) {
      timing_render_queued = timing_render_queued || force || fingerprint !== timing_fingerprint;
      return;
    }
    timing_render_in_flight = true;
    try {
      await processing_render_update_timing_before_stability();
      timing_fingerprint = fingerprint;
    } finally {
      timing_render_in_flight = false;
      if (timing_render_queued) {
        timing_render_queued = false;
        queueMicrotask(() => processing_render_update_timing(true));
      }
    }
  };

  // The operations legend is intentionally small. Family/layer/kernel identity
  // stays in hover rather than becoming dozens of legend entries.
  const processing_render_timeline_before_legend_polish = processing_render_timeline;
  processing_render_timeline = async function(payload) {
    await processing_render_timeline_before_legend_polish(payload);
    const mount = by_id("processing_timeline_plot");
    if (!mount || !Array.isArray(mount.data)) return;
    const seen = new Set();
    const updates = [];
    mount.data.forEach((trace, index) => {
      const name = String(trace.name || "");
      const parts = name.trim().split(/\s+/);
      const owner = parts.shift() || "";
      const operation = parts.join(" ") || "other";
      const compact = `${owner} ${operation}`.trim();
      const showlegend = !seen.has(compact);
      if (showlegend) seen.add(compact);
      updates.push(Plotly.restyle(mount, {name:compact, showlegend}, [index]));
    });
    await Promise.all(updates);
  };

  const processing_render_before_stability_polish = processing_render;
  processing_render = async function(payload, trace_available) {
    await processing_render_before_stability_polish(payload, trace_available);
    if (app.processing_pair_state_final_installed !== true) {
      apply_processing_pair(payload, Boolean(trace_available));
    }
    polish_processing_labels();
    processing_gpu_link_time_axes();
  };

  window.addEventListener("load", () => {
    polish_processing_labels();
    decorate_paired_eyes();
  });
})();
// ^^^ THOG
