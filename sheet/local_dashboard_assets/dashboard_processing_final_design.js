// vvv THOG
"use strict";

(function install_processing_final_design() {
  const style_id = "instra-processing-final-design-style";
  if (!document.getElementById(style_id)) {
    const style = document.createElement("style");
    style.id = style_id;
    style.textContent = `
      #processing_timeline_card:not(.maximized) {
        min-height:190px !important;
        height:190px !important;
      }
      #processing_timeline_card:not(.maximized) .processing-plot-shell {
        min-height:118px !important;
        height:118px !important;
        flex:0 0 118px !important;
      }
      #processing_timeline_card .chart-card-header {
        min-height:54px !important;
        height:auto !important;
      }
      .processing-hard-constraints-button {
        height:27px;
        padding:0 8px;
        border:1px solid rgba(127,127,127,.32);
        border-radius:4px;
        background:#f7f7f8;
        color:inherit;
        cursor:pointer;
        font-size:10px;
        font-weight:400;
        white-space:nowrap;
      }
      .processing-hard-constraints-button:hover,
      .processing-hard-constraints-button[aria-pressed="true"] {
        background:#eceafc;
        border-color:#b8afea;
        color:#4732b7;
      }
      .processing-hard-constraints-panel {
        padding:6px 12px 12px;
        overflow:auto;
      }
      .processing-hard-constraints-table {
        width:100%;
        border-collapse:collapse;
        font-size:11px;
        font-variant-numeric:tabular-nums;
      }
      .processing-hard-constraints-table th,
      .processing-hard-constraints-table td {
        padding:5px 8px;
        border-bottom:1px solid rgba(127,127,127,.18);
        text-align:right;
        white-space:nowrap;
      }
      .processing-hard-constraints-table th:first-child,
      .processing-hard-constraints-table td:first-child { text-align:left; }
      .processing-hard-constraints-table tr.limiting td:first-child { font-weight:750; }
      .processing-hard-constraints-table .over-capacity { color:#c62828; font-weight:750; }
      .processing-hard-constraints-note {
        margin:6px 0 0;
        color:#69717d;
        font-size:10px;
      }
    `;
    document.head.appendChild(style);
  }

  // The user now prefers every sampled NSYS resource visible initially. Legend
  // clicks still hide individual traces and Reset returns to this all-visible state.
  if (Array.isArray(processing_gpu_resource_specs)) {
    processing_gpu_resource_specs.forEach(spec => { spec.visible = true; });
  }

  function strip_chart_letters(text) {
    return String(text || "").replace(/^\s*[ABC]\.\s*/i, "");
  }

  function enforce_titles() {
    const titles = [
      ["processing_resource_card", "Stream Resource Contention"],
      ["processing_timeline_card", "GPT Level Operations by Stream (MAIN/PREMAT)"],
      ["processing_compatibility_card", "PREMAT Compatibility"],
    ];
    for (const [card_id, title] of titles) {
      const heading = by_id(card_id)?.querySelector(".chart-heading-copy h2");
      if (heading && heading.textContent !== title) heading.textContent = title;
    }
    if (typeof chart_titles === "object") {
      chart_titles.processing_resource = "Stream Resource Contention";
      chart_titles.processing_timeline = "GPT Level Operations by Stream (MAIN/PREMAT)";
      chart_titles.processing_compatibility = "PREMAT Compatibility";
    }
    for (const heading of document.querySelectorAll("#processing_chart_group .chart-heading-copy h2")) {
      const stripped = strip_chart_letters(heading.textContent);
      if (stripped !== heading.textContent) heading.textContent = stripped;
    }
  }

  // Compact interval bars, not long scatter-line "snakes". The two owner bands
  // stay because simultaneous PREMAT work is real information, but the card is
  // deliberately a narrow semantic timeline rather than a large plotting field.
  function semantic_label(row) {
    const operation = String(row.operation || "misc").toLowerCase();
    const family = String(row.family || "").toUpperCase();
    if (operation === "materialize") return `${family || "?"} MAT`;
    if (operation === "consume") return `${family || "?"} GEMM`;
    return operation === "lm_head" ? "LM HEAD" : operation.toUpperCase();
  }

  processing_render_timeline = async function(payload) {
    const intervals = Array.isArray(payload.intervals) ? payload.intervals : [];
    const lane_order = ["MAIN", "PREMAT", "OTHER", "UNKNOWN"];
    const owners = lane_order.filter(owner => intervals.some(row => String(row.owner || "UNKNOWN") === owner));
    const groups = new Map();
    for (const row of intervals) {
      const owner = String(row.owner || "UNKNOWN");
      const label = semantic_label(row);
      const key = `${owner}:${label}`;
      if (!groups.has(key)) groups.set(key, {owner, label, rows:[]});
      groups.get(key).rows.push(row);
    }

    const traces = [];
    for (const group of groups.values()) {
      const operation = String(group.rows[0]?.operation || "misc");
      traces.push({
        type:"bar",
        orientation:"h",
        name:`${group.owner} ${group.label}`,
        x:group.rows.map(row => Math.max(0, Number(row.end_us) - Number(row.start_us)) / 1000.0),
        base:group.rows.map(row => Number(row.start_us) / 1000.0),
        y:group.rows.map(() => group.owner),
        width:group.owner === "PREMAT" ? 0.46 : 0.66,
        marker:{
          color:processing_gpu_operation_colour(operation, group.owner),
          line:operation === "materialize" ? {color:"#4a0030", width:0.8} : {width:0},
        },
        customdata:group.rows.map(row => [
          semantic_label(row),
          row.family || "",
          row.layer === "" || row.layer === null || row.layer === undefined ? "—" : Number(row.layer) + 1,
          row.kernel_name || "",
        ]),
        hovertemplate:"%{y} · %{customdata[0]} · L%{customdata[2]}<br>%{customdata[3]}<br>%{x:.4f} ms<extra></extra>",
      });
    }
    const capture_ms = Number(payload.metadata?.capture_duration_ms || 0);
    await processing_plot("processing_timeline_plot", traces, {
      margin:{l:88, r:34, t:8, b:36},
      barmode:"overlay",
      hovermode:"closest",
      legend:{orientation:"h", y:1.14, font:{size:9}},
      xaxis:{title:"capture time (ms)", range:capture_ms > 0 ? [0, capture_ms] : undefined},
      yaxis:{categoryorder:"array", categoryarray:owners, automargin:true, fixedrange:true},
    });
    processing_gpu_link_time_axes();
  };

  function config_value(run, ...names) {
    const config = run?.configuration && typeof run.configuration === "object" ? run.configuration : {};
    for (const name of names) {
      const value = config[name] ?? run?.[name];
      if (value !== undefined && value !== null && value !== "") return value;
    }
    return null;
  }

  function normalized_value(value) {
    if (typeof value === "boolean") return value;
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    if (Number.isFinite(number) && String(value).trim() !== "") return number;
    return String(value).trim().toLowerCase();
  }

  function comparison_signature(run) {
    return {
      host:normalized_value(config_value(run, "host_label")),
      layers:normalized_value(config_value(run, "n_layer")),
      heads:normalized_value(config_value(run, "n_head")),
      d_model:normalized_value(config_value(run, "n_embd")),
      context:normalized_value(config_value(run, "block_size")),
      batch:normalized_value(config_value(run, "batch_size")),
      accumulation:normalized_value(config_value(run, "gradient_accumulation_steps")),
      optimizer:normalized_value(config_value(run, "optimizer_name", "optimizer")),
      learning_rate:normalized_value(config_value(run, "learning_rate")),
      min_learning_rate:normalized_value(config_value(run, "min_learning_rate", "min_lr")),
      warmup:normalized_value(config_value(run, "warmup_iters", "warmup_steps", "warmup_updates")),
      dtype:normalized_value(config_value(run, "dtype")),
      backend:normalized_value(config_value(run, "attention_backend", "backend")),
      activation_checkpointing:normalized_value(config_value(run, "activation_checkpointing")),
      checkpoint_segment:normalized_value(config_value(run, "checkpoint_recompute_segment")),
    };
  }

  function signatures_match(left, right) {
    const a = comparison_signature(left);
    const b = comparison_signature(right);
    const essential = ["host", "layers", "heads", "d_model", "context", "batch", "accumulation"];
    if (essential.some(key => a[key] === null || b[key] === null || a[key] !== b[key])) return false;
    return Object.keys(a).every(key => a[key] === b[key]);
  }

  const processing_throughput_workspace_runs_before_final_design = processing_throughput_workspace_runs;
  processing_throughput_workspace_runs = function() {
    if (app.workspace_mode === true) return processing_throughput_workspace_runs_before_final_design();
    const selected = typeof current_run === "function" ? current_run() : null;
    if (!selected) return [];
    const selected_id = String(run_identifier(selected));
    const runs = (app.runs || []).filter(run => {
      const run_id = String(run_identifier(run));
      return run_id === selected_id || (is_visible(run_id) && signatures_match(selected, run));
    });
    if (!runs.some(run => String(run_identifier(run)) === selected_id)) runs.unshift(selected);
    return runs;
  };

  if (typeof processing_update_timing_pair_assessment === "function") {
    const prior_assessment = processing_update_timing_pair_assessment;
    processing_update_timing_pair_assessment = function(entries) {
      if (app.workspace_mode === true) return prior_assessment(entries);
      if (!entries.length) return {level:"info", text:"No complete-update timing data for this comparison cohort."};
      if (entries.length === 1) return {level:"info", text:"Open the eye on another matching run to add it to comparable timing charts."};
      return {level:"ok", text:`Matched comparison cohort · ${entries.length} visible runs with compatible execution controls.`};
    };
  }

  processing_view.show_hard_constraints = Boolean(processing_view.show_hard_constraints);
  processing_view.final_payload = null;

  function hard_constraint_panel() {
    let panel = by_id("processing_hard_constraints_panel");
    const card = by_id("processing_compatibility_card");
    if (!card) return null;
    if (!panel) {
      panel = document.createElement("div");
      panel.id = "processing_hard_constraints_panel";
      panel.className = "processing-hard-constraints-panel";
      panel.hidden = true;
      card.appendChild(panel);
    }
    return panel;
  }

  function ensure_hard_constraint_button(payload) {
    const card = by_id("processing_compatibility_card");
    const actions = card?.querySelector(".chart-card-actions");
    if (!actions) return;
    let button = by_id("processing_hard_constraints_button");
    if (!button) {
      button = document.createElement("button");
      button.id = "processing_hard_constraints_button";
      button.type = "button";
      button.className = "processing-hard-constraints-button";
      button.textContent = "Show Just Hard Constraints";
      button.addEventListener("click", () => {
        processing_view.show_hard_constraints = !processing_view.show_hard_constraints;
        render_hard_constraints(processing_view.final_payload || {});
      });
      actions.insertBefore(button, actions.firstChild);
    }
    const rows = Array.isArray(payload?.premat_hard_constraints) ? payload.premat_hard_constraints : [];
    button.hidden = rows.length === 0;
  }

  function format_hard_value(value, unit) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    if (unit === "bytes") {
      if (Math.abs(number) >= 1024 * 1024) return `${(number / (1024 * 1024)).toFixed(2)} MiB`;
      if (Math.abs(number) >= 1024) return `${(number / 1024).toFixed(1)} KiB`;
      return `${number.toFixed(0)} B`;
    }
    return number.toLocaleString(undefined, {maximumFractionDigits:1});
  }

  function render_hard_constraints(payload) {
    const card = by_id("processing_compatibility_card");
    const panel = hard_constraint_panel();
    const button = by_id("processing_hard_constraints_button");
    const plot_shell = card?.querySelector(":scope > .processing-plot-shell");
    const key = card?.querySelector(".processing-compatibility-key");
    if (!card || !panel || !button) return;
    const rows = Array.isArray(payload?.premat_hard_constraints) ? payload.premat_hard_constraints : [];
    if (!rows.length) {
      processing_view.show_hard_constraints = false;
      panel.hidden = true;
      if (plot_shell) plot_shell.hidden = false;
      if (key) key.hidden = false;
      button.hidden = true;
      return;
    }
    button.hidden = false;
    button.setAttribute("aria-pressed", processing_view.show_hard_constraints ? "true" : "false");
    button.textContent = processing_view.show_hard_constraints ? "Show Compatibility Strip" : "Show Just Hard Constraints";
    if (!processing_view.show_hard_constraints) {
      panel.hidden = true;
      if (plot_shell) plot_shell.hidden = false;
      if (key) key.hidden = false;
      processing_resize_ready_card(card);
      return;
    }
    if (plot_shell) plot_shell.hidden = true;
    if (key) key.hidden = true;
    panel.hidden = false;
    const stage = rows[0] || {};
    panel.innerHTML = `
      <p class="processing-hard-constraints-stage"><strong>${processing_escape(stage.compatibility_class || "—")}</strong> · most constrained PREMAT stage ${Number(stage.premat_stage_index || 1)}/${Number(stage.premat_stage_count || 1)} · ${processing_escape(stage.premat_kernel_name || "unknown kernel")}</p>
      <table class="processing-hard-constraints-table">
        <thead><tr><th>Hard constraint</th><th>SM capacity</th><th>MAIN / block</th><th>PREMAT / block</th><th>Full MAIN + 1 PREMAT</th><th>Capacity used</th></tr></thead>
        <tbody>${rows.map(row => {
          const pct = Number(row.full_main_plus_one_pct);
          const over = Number.isFinite(pct) && pct > 100;
          return `<tr class="${row.limiting ? "limiting" : ""}">
            <td>${processing_escape(row.resource)}${row.limiting ? " · limiter" : ""}</td>
            <td>${processing_escape(format_hard_value(row.capacity, row.unit))}</td>
            <td>${processing_escape(format_hard_value(row.main_per_block, row.unit))}</td>
            <td>${processing_escape(format_hard_value(row.premat_per_block, row.unit))}</td>
            <td>${processing_escape(format_hard_value(row.full_main_plus_one_premat, row.unit))}</td>
            <td class="${over ? "over-capacity" : ""}">${Number.isFinite(pct) ? `${pct.toFixed(1)}%` : "—"}</td>
          </tr>`;
        }).join("")}</tbody>
      </table>
      <p class="processing-hard-constraints-note">The five hard SM-residency limits only. “Full MAIN + 1 PREMAT” tests one PREMAT block against MAIN at its theoretical full block residency; this is structural NCU evidence, not a time-resolved NSYS counter.</p>`;
  }

  function apply_final_design(payload) {
    enforce_titles();
    ensure_hard_constraint_button(payload);
    render_hard_constraints(payload);
  }

  const processing_render_before_final_design = processing_render;
  processing_render = async function(payload, trace_available) {
    processing_view.final_payload = payload;
    await processing_render_before_final_design(payload, trace_available);
    processing_view.final_payload = payload;
    apply_final_design(payload);
    requestAnimationFrame(() => apply_final_design(payload));
  };

  window.addEventListener("load", () => {
    enforce_titles();
    const group = by_id("processing_chart_group");
    if (group && typeof MutationObserver === "function") {
      const observer = new MutationObserver(() => enforce_titles());
      observer.observe(group, {childList:true, subtree:true, characterData:true});
    }
  });
})();
// ^^^ THOG
