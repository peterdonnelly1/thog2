// vvv THOG
"use strict";

/*
 * Focused Runs-table restoration.
 *
 * This extracts only the established run-summary behaviour that was lost when
 * the broad legacy Workspace bundle was removed: the hyperparameter/resource
 * columns, one canonical column order, and the draggable RUN NAME width.  It
 * deliberately does not own Workspace, startup policy, charts or navigation.
 */
window.addEventListener("load", () => {
  setTimeout(() => {
    const name_width_storage_key = "thog2_local_run_name_column_width";
    const default_name_width = 390;

    const finite_number = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };
    const clamp_integer = (value, minimum, maximum) => Math.max(
      minimum,
      Math.min(maximum, Math.round(Number(value))),
    );
    const configuration = run => (
      run?.configuration && typeof run.configuration === "object" && !Array.isArray(run.configuration)
        ? run.configuration
        : {}
    );
    const configured_value = (run, ...names) => {
      const config = configuration(run);
      const lifecycle = config.lifecycle && typeof config.lifecycle === "object" ? config.lifecycle : {};
      for (const name of names) {
        if (run?.[name] !== undefined && run[name] !== null && run[name] !== "") return run[name];
        if (config?.[name] !== undefined && config[name] !== null && config[name] !== "") return config[name];
        if (lifecycle?.[name] !== undefined && lifecycle[name] !== null && lifecycle[name] !== "") return lifecycle[name];
      }
      return null;
    };
    const display_value = value => {
      if (value === null || value === undefined || value === "") return "—";
      const numeric = finite_number(value);
      return numeric === null ? String(value) : String(Math.round(numeric * 10) / 10);
    };
    const display_boolean = value => {
      if (value === true || value === 1 || String(value).toLowerCase() === "true") return "Y";
      if (value === false || value === 0 || String(value).toLowerCase() === "false") return "N";
      return "—";
    };
    const optimizer_text = run => {
      const raw = configured_value(run, "optimizer_name", "optimizer");
      if (!raw) return "—";
      const name = String(raw).toLowerCase();
      const momentum = finite_number(configured_value(run, "optimizer_momentum"));
      return ["sgd", "sgd_nesterov", "rmsprop"].includes(name) && momentum !== null && momentum > 0
        ? `${name}_${momentum}`
        : name;
    };
    const learning_rate_code = (run, ...names) => {
      const value = finite_number(configured_value(run, ...names));
      return value !== null && value >= 0 ? String(Number((value * 100000).toFixed(8))) : "—";
    };
    const parameter_report = run => {
      const report = configuration(run).parameter_report;
      return report && typeof report === "object" && !Array.isArray(report) ? report : {};
    };
    const million_parameters = (run, dense_equivalent) => {
      const report = parameter_report(run);
      const raw = dense_equivalent
        ? (report.dense_equivalent_total_parameters ?? report.dense_equivalent_parameters)
        : (report.persistent_parameters ?? report.trainable_parameters);
      const value = finite_number(raw);
      return value === null ? "—" : String(Math.round(value / 1_000_000));
    };
    const gpu_value = run => {
      if (run?.gpu_assignment) return run.gpu_assignment.status === "verified" ? run.gpu_assignment.ordinal : "?";                             // <<< THOG never display a guessed GPU as verified
      const explicit = configured_value(run, "gpu_index");
      if (explicit !== null) return explicit;
      const source = [
        configuration(run).cuda_visible_devices,
        configuration(run).command,
        run?.command,
        run?.host_label,
      ].filter(Boolean).join(" ");
      const environment_match = source.match(/CUDA_VISIBLE_DEVICES\s*=\s*["']?(\d+)/i);
      if (environment_match) return environment_match[1];
      const host_match = source.match(/(?:^|[_ -])gpu[_ -]?(\d+)(?:$|[_ -])/i);
      if (host_match) return host_match[1];
      return /(?:^|[_ -])scruffy(?:$|[_ -])/i.test(source) ? 0 : null;
    };
    const premat_matrix_text = run => {
      const premat = configured_value(run, "premat");
      const enabled = premat === true || premat === 1 || String(premat || "").toLowerCase() === "enabled";
      if (!enabled) return "—";
      const raw = configured_value(run, "premat_target_matrix");
      let selected;
      if (raw === null || raw === undefined || raw === "") {
        selected = new Set([1, 2, 3, 4]);
      } else if (Array.isArray(raw)) {
        selected = new Set(raw.map(Number));
      } else {
        selected = new Set((String(raw).match(/[1-4]/g) || []).map(Number));
      }
      return [1, 2, 3, 4].map(matrix => selected.has(matrix) ? String(matrix) : " ").join(" ");
    };

    const generated = Object.freeze({
      gpu: {label: "GPU", title: "physical GPU selected for this run", numeric: true, value: gpu_value},
      preset: {
        label: "p", title: "preset", numeric: false,
        value: run => run?.preset || configured_value(run, "geometry_preset", "model_type"),
      },
      optimizer: {label: "OPT", title: "optimizer (momentum suffix when used)", numeric: false, value: optimizer_text},
      gb: {
        label: "GB", title: "peak process GPU memory allocated so far (GiB)", numeric: true,
        value: run => configured_value(run, "gpu_peak_memory_allocated_gb"),
      },
      layers: {label: "L", title: "layers", numeric: true, value: run => configured_value(run, "n_layer")},
      depth_order: {
        label: "P", title: "depth order (not applicable to DENSE)", numeric: true,
        value: run => String(run?.model_type || configured_value(run, "model_type") || "").toLowerCase() === "dense"
          ? "—"
          : configured_value(run, "o_depth"),
      },
      premat: {label: "premat", title: "prematerialised matrix numbers (1=QKV, 2=O, 3=UP, 4=DOWN)", numeric: false, value: premat_matrix_text},
      parms: {label: "PARMS", title: "persistent model parameters (millions, rounded)", numeric: true, value: run => million_parameters(run, false)},
      equiv: {
        label: "EQUIV", title: "dense-equivalent parameters (millions, THOG only)", numeric: true,
        value: run => String(run?.model_type || configured_value(run, "model_type") || "").toLowerCase() === "dense"
          ? "—"
          : million_parameters(run, true),
      },
      warmup: {label: "w", title: "warm-up steps", numeric: true, value: run => configured_value(run, "warmup_iters", "warmup_steps", "warmup_updates")},
      context: {label: "C", title: "context length", numeric: true, value: run => configured_value(run, "block_size")},
      d_model: {label: "D", title: "d_model", numeric: true, value: run => configured_value(run, "n_embd")},
      heads: {label: "H", title: "attention heads", numeric: true, value: run => configured_value(run, "n_head")},
      grad_accum: {label: "A", title: "gradient accumulation steps", numeric: true, value: run => configured_value(run, "gradient_accumulation_steps")},
      activation_checkpointing: {
        label: "S", title: "activation checkpointing", numeric: true,
        value: run => display_boolean(configured_value(run, "activation_checkpointing")),
      },
      learning_rate: {label: "c", title: "maximum learning rate in units of 1e-5", numeric: true, value: run => learning_rate_code(run, "learning_rate")},
      min_learning_rate: {label: "f", title: "minimum learning rate in units of 1e-5", numeric: true, value: run => learning_rate_code(run, "min_learning_rate", "min_lr")},
      capture_period: {
        label: "C_p", title: "weight-curve capture period in optimizer steps", numeric: true,
        value: run => configured_value(run, "instrumentation__depth_weight_curves__log_every_n_steps")
          ?? configuration(run).instrumentation_configuration?.THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_LOG_EVERY_N_STEPS,
      },
    });

    const order = Object.freeze([
      "select", "visibility", "steps", "duration", "state", "name", "wandb", "host", "gpu",
      "preset", "optimizer", "gb", "layers", "depth_order", "premat", "parms", "equiv", "warmup",
      "context", "d_model", "heads", "grad_accum", "activation_checkpointing", "learning_rate",
      "min_learning_rate", "probe_start", "probe_end", "curve_start", "curve_end", "capture_period",
      "updated", "menu",
    ]);
    const widths = Object.freeze({
      select:34, visibility:34, steps:62, duration:72, state:88, wandb:0, host:84, gpu:42,
      preset:64, optimizer:70, gb:52, layers:42, depth_order:42, premat:76, parms:58, equiv:58, warmup:42,
      context:64, d_model:64, heads:42, grad_accum:46, activation_checkpointing:42,
      learning_rate:42, min_learning_rate:42, probe_start:50, probe_end:50,
      curve_start:50, curve_end:64, capture_period:50, updated:92, menu:36,
    });

    const tag = (element, key) => {
      if (element) element.dataset.instraColumnKey = key;
      return element;
    };
    const tag_base_headers = () => {
      const row = document.querySelector(".runs-table thead tr");
      if (!row) return;
      const by_text = text => [...row.children].find(cell => String(cell.textContent || "").trim().toUpperCase() === text);
      tag(row.querySelector(".check-column"), "select");
      tag(row.querySelector(".visibility-column"), "visibility");
      tag(row.querySelector(".name-column"), "name");
      tag(by_text("W&B ID"), "wandb");
      tag(by_text("STATE"), "state");
      tag(by_text("HOST"), "host");
      tag(row.querySelector(".probe-start-column"), "probe_start");
      tag(row.querySelector(".probe-end-column"), "probe_end");
      tag(row.querySelector(".curve-start-column"), "curve_start");
      tag(row.querySelector(".curve-end-column"), "curve_end");
      tag(row.querySelector(".step-column"), "steps");
      tag(row.querySelector(".duration-column"), "duration");
      tag(by_text("UPDATED"), "updated");
      tag(row.querySelector(".menu-column"), "menu");
    };
    const tag_base_row = row => {
      const keys = [
        "select", "visibility", "name", "wandb", "state", "host",
        "probe_start", "probe_end", "curve_start", "curve_end",
        "steps", "duration", "updated", "menu",
      ];
      [...row.children].slice(0, keys.length).forEach((cell, index) => tag(cell, keys[index]));
    };

    function ensure_generated_headers() {
      const row = document.querySelector(".runs-table thead tr");
      if (!row) return;
      for (const [key, definition] of Object.entries(generated)) {
        let header = row.querySelector(`[data-instra-run-summary-header="${key}"]`);
        if (!header) {
          header = document.createElement("th");
          header.dataset.instraRunSummaryHeader = key;
          header.className = definition.numeric ? "numeric-column instra-run-summary-column" : "instra-run-summary-column";
          header.textContent = definition.label;
          header.title = definition.title;
          row.appendChild(header);
        }
        tag(header, key);
      }
    }

    function ensure_generated_cells(row, run) {
      for (const [key, definition] of Object.entries(generated)) {
        let cell = row.querySelector(`[data-instra-run-summary-cell="${key}"]`);
        if (!cell) {
          cell = document.createElement("td");
          cell.dataset.instraRunSummaryCell = key;
          cell.className = definition.numeric ? "numeric-column instra-run-summary-column" : "instra-run-summary-column";
          row.appendChild(cell);
        }
        const raw = definition.value(run);
        const shown = definition.key === "activation_checkpointing" ? raw : display_value(raw);
        cell.textContent = shown;
        cell.title = key === "gpu" && run?.gpu_assignment
          ? `${run.gpu_assignment.status}: recorded GPU ${run.gpu_assignment.recorded_ordinal ?? run.gpu_assignment.ordinal ?? "unknown"}; ${run.gpu_assignment.model || "model unknown"} ${run.gpu_assignment.uuid || ""}`
          : `${definition.title}: ${shown}`;                                                                                                                 // <<< THOG expose verified GPU model and stable UUID
        tag(cell, key);
        if (key === "preset") {
          cell.classList.toggle("instra-dense-preset", String(shown).trim().toLowerCase() === "dense");
        }
      }
    }

    const reorder = row => {
      const elements = new Map(
        [...row.children]
          .filter(element => element.dataset.instraColumnKey)
          .map(element => [element.dataset.instraColumnKey, element]),
      );
      const ordered = order.map(key => elements.get(key)).filter(Boolean);
      const remainder = [...row.children].filter(element => !ordered.includes(element));
      const fragment = document.createDocumentFragment();
      [...ordered, ...remainder].forEach(element => fragment.appendChild(element));
      row.appendChild(fragment);
    };

    const stored_name_width = () => {
      const value = finite_number(localStorage.getItem(name_width_storage_key));
      return value === null ? default_name_width : clamp_integer(value, 180, 2200);
    };
    const apply_geometry = requested_name_width => {
      const table = document.querySelector(".runs-table");
      const header_row = table?.querySelector("thead tr");
      if (!table || !header_row) return;
      const name_width = clamp_integer(requested_name_width, 180, 2200);
      let fixed_width = 0;
      for (const header of header_row.children) {
        const key = header.dataset.instraColumnKey;
        if (!key) continue;
        if (key === "name") {
          header.style.setProperty("width", "auto", "important");
          header.style.setProperty("min-width", "0", "important");
          header.style.setProperty("max-width", "none", "important");
          continue;
        }
        const width = widths[key];
        if (width === undefined) continue;
        fixed_width += width;
        header.style.setProperty("width", `${width}px`, "important");
        header.style.setProperty("min-width", `${width}px`, "important");
        header.style.setProperty("max-width", `${width}px`, "important");
      }
      table.style.setProperty("--instra-run-name-width", `${name_width}px`);
      table.style.setProperty("width", "100%", "important");
      table.style.setProperty("min-width", `${fixed_width + name_width}px`, "important");
    };
    const set_name_width = (width, persist = false) => {
      const resolved = clamp_integer(width, 180, 2200);
      if (persist) localStorage.setItem(name_width_storage_key, String(resolved));
      apply_geometry(resolved);
    };
    const install_name_resizer = () => {
      const header = document.querySelector('.runs-table th[data-instra-column-key="name"]');
      if (!header) return;
      let handle = header.querySelector(".run-name-column-resizer");
      if (handle?.dataset.instraOwner === "focused-restore") return;
      handle?.remove();
      handle = document.createElement("span");
      handle.className = "run-name-column-resizer";
      handle.dataset.instraOwner = "focused-restore";
      handle.setAttribute("role", "separator");
      handle.setAttribute("aria-orientation", "vertical");
      handle.setAttribute("aria-label", "Resize RUN NAME column");
      handle.title = "Drag to resize RUN NAME; double-click to reset";
      handle.addEventListener("pointerdown", event => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();
        const start_x = event.clientX;
        const start_width = header.getBoundingClientRect().width;
        handle.classList.add("dragging");
        const move = pointer_event => set_name_width(start_width + pointer_event.clientX - start_x);
        const finish = pointer_event => {
          handle.classList.remove("dragging");
          window.removeEventListener("pointermove", move, true);
          window.removeEventListener("pointerup", finish, true);
          set_name_width(start_width + pointer_event.clientX - start_x, true);
        };
        window.addEventListener("pointermove", move, true);
        window.addEventListener("pointerup", finish, true);
      });
      handle.addEventListener("dblclick", event => {
        event.preventDefault();
        event.stopPropagation();
        localStorage.removeItem(name_width_storage_key);
        set_name_width(default_name_width);
      });
      header.appendChild(handle);
    };

    const polish = () => {
      const table = document.querySelector(".runs-table");
      const header_row = table?.querySelector("thead tr");
      if (!table || !header_row) return;
      tag_base_headers();
      ensure_generated_headers();
      reorder(header_row);
      for (const row of table.querySelectorAll("tbody tr[data-run-id]")) {
        const run = (app.runs || []).find(candidate => String(run_identifier(candidate)) === String(row.dataset.runId));
        if (!run) continue;
        tag_base_row(row);
        ensure_generated_cells(row, run);
        reorder(row);
      }
      install_name_resizer();
      apply_geometry(stored_name_width());
      const column_count = header_row.children.length;
      table.querySelectorAll("tbody .group-row td").forEach(cell => { cell.colSpan = column_count; });
    };

    const base_append_run_row_restore = append_run_row;
    append_run_row = function(body, run) {
      const result = base_append_run_row_restore(body, run);
      const row = body.lastElementChild;
      if (row?.matches?.("tr[data-run-id]")) {
        tag_base_row(row);
        ensure_generated_cells(row, run);
      }
      return result;
    };

    const base_render_runs_restore = render_runs;
    render_runs = function() {
      const result = base_render_runs_restore();
      polish();
      return result;
    };

    const style = document.createElement("style");
    style.id = "instra-focused-runs-table-restore-style";
    style.textContent = `
      .runs-table .instra-run-summary-column { white-space:nowrap; font-variant-numeric:tabular-nums; }
      .runs-table [data-instra-column-key="wandb"] { display:none !important; }
      .runs-table .instra-dense-preset { font-weight:750 !important; }
      .runs-table [data-instra-column-key="layers"],
      .runs-table [data-instra-column-key="depth_order"] { color:#7a1f3d !important; font-weight:700; }
      .runs-table [data-instra-column-key="premat"] {
        white-space:pre !important; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
        font-variant-numeric:tabular-nums; text-align:left !important; text-transform:none !important;
      }
      .runs-table .name-column {
        position:relative; width:auto !important; min-width:0 !important; max-width:none !important;
      }
      .run-name-column-resizer {
        position:absolute; z-index:5; top:0; right:-4px; bottom:0;
        width:9px; cursor:col-resize; touch-action:none;
      }
      .run-name-column-resizer::after {
        content:""; position:absolute; top:5px; right:4px; bottom:5px;
        width:1px; background:#b8bec6;
      }
      .run-name-column-resizer:hover::after,
      .run-name-column-resizer.dragging::after { width:2px; background:#1590a8; }
    `;
    document.head.appendChild(style);

    polish();
    render_runs();
  }, 120);
});
// ^^^ THOG
