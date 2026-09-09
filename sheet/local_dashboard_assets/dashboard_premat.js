// vvv THOG completed-microstep Premat playback and per-matrix outcome inspector
"use strict";

const PREMAT_FINAL_HOLD_MS = 1000;
const PREMAT_PREMATERIALISING_DURATION_MULTIPLIER = 1.5;
const PREMAT_STATE_CLASSES = [
  "premat-neutral",
  "premat-pending",
  "premat-state-materialising",
  "premat-state-available",
  "premat-state-consuming-full",
  "premat-state-consumed-full",
  "premat-state-waiting",
  "premat-state-consuming-waited",
  "premat-state-consumed-waited",
  "premat-state-main-materialising",
  "premat-state-main-consuming",
  "premat-state-consumed-main",
];

const premat_view = {
  run_id: null,
  request_serial: 0,
  latest_received_update: 0,
  pending_snapshot: null,
  active_snapshot: null,
  active_model: null,
  frames: [],
  frame_index: 0,
  playback_running: false,
  playing: true,
  playback_timer: null,
  state_duration_ms: 250,
  cell_elements: new Map(),
  row_elements: new Map(),
  active_layer_index: null,
};

function premat_escape(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[character]));
}

function premat_bytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes)) return "—";
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}

function premat_ms(value) {
  const milliseconds = Number(value);
  return Number.isFinite(milliseconds) ? `${milliseconds.toFixed(3)} ms` : "—";
}

function premat_matrix_size(value) {
  const mebibytes = Number(value) / (1024 ** 2);
  if (!Number.isFinite(mebibytes)) return "—";
  return `${mebibytes >= 10 ? mebibytes.toFixed(0) : mebibytes.toFixed(1)} MiB`;
}

function premat_memory_rule(snapshot) {
  if (snapshot?.headroom_mode === "stay_within_global_buffer") {
    const bytes = Number(snapshot?.memory?.global_buffer_bytes);
    const gib = Number.isFinite(bytes)
      ? (bytes / 1024 ** 3).toFixed(3).replace(/0+$/, "").replace(/\.$/, "")
      : "—";
    return `rule: do not cross global buffer - currently ${gib} GiB`;
  }
  return "rule: stay below current peak memory";
}

function premat_candidate_key(layer_index, family) {
  return `${Number(layer_index)}:${String(family)}`;
}

function premat_families(attention_mode) {
  return attention_mode === "unfused"
    ? ["QK", "V", "O", "UP", "DOWN"]
    : ["QKV", "O", "UP", "DOWN"];
}

function premat_family_label(family, attention_mode) {
  const labels = attention_mode === "unfused"
    ? {QK: "ATTN UNFUSED · QK", V: "ATTN UNFUSED · V", O: "ATTN O", UP: "MLP UP", DOWN: "MLP DN"}
    : {QKV: "ATTN FUSED · QKV", O: "ATTN O", UP: "MLP UP", DOWN: "MLP DN"};
  return labels[family] || String(family);
}

function premat_stages(attention_mode) {
  return attention_mode === "unfused"
    ? [
        ["LN1", null], ["ATTN UNFUSED · QK", "QK"], ["score", null],
        ["scale / mask", null], ["softmax", null], ["ATTN UNFUSED · V", "V"],
        ["attention", null], ["ATTN O", "O"], ["resid", null], ["LN2", null],
        ["MLP UP", "UP"], ["GELU", null], ["MLP DN", "DOWN"], ["resid", null],
      ]
    : [
        ["LN1", null], ["ATTN FUSED · QKV", "QKV"], ["attention", null],
        ["ATTN O", "O"], ["resid", null], ["LN2", null],
        ["MLP UP", "UP"], ["GELU", null], ["MLP DN", "DOWN"], ["resid", null],
      ];
}

function premat_layer_indices(snapshot) {
  const declared = Array.isArray(snapshot?.layer_indices) ? snapshot.layer_indices : [];
  const observed = (snapshot?.events || []).map(event => event.layer_index);
  return [...new Set([...declared, ...observed].map(Number).filter(Number.isFinite))]
    .sort((left, right) => left - right);
}

function premat_new_record(layer_index, family, attention_mode) {
  return {
    key: premat_candidate_key(layer_index, family),
    layer_index,
    family,
    label: premat_family_label(family, attention_mode),
    trace: [],
    outcome: "NOT YET REACHED",
    path: "none",
    admission_reason: "not checked",
    wait_ms: null,
    materialisation_ms: null,
    main_materialisation_ms: null,
    completion_at_deadline_percent: null,
    retained_bytes: null,
  };
}

function premat_append_trace(record, state) {
  if (record.trace[record.trace.length - 1] !== state) record.trace.push(state);
}

function premat_update_from_event(record, event) {
  if (event.predicted_retained_bytes !== null
      && event.predicted_retained_bytes !== undefined
      && Number.isFinite(Number(event.predicted_retained_bytes))) {
    record.retained_bytes = Number(event.predicted_retained_bytes);
  }
  if (event.admission_reason) record.admission_reason = String(event.admission_reason).replaceAll("_", " ");
  if (event.wait_ms !== null && event.wait_ms !== undefined && Number.isFinite(Number(event.wait_ms))) {
    record.wait_ms = Number(event.wait_ms);
  }
  if (event.materialisation_ms !== null && event.materialisation_ms !== undefined && Number.isFinite(Number(event.materialisation_ms))) {
    record.materialisation_ms = Number(event.materialisation_ms);
  }
  if (event.main_stream_materialisation_ms !== null
      && event.main_stream_materialisation_ms !== undefined
      && Number.isFinite(Number(event.main_stream_materialisation_ms))) {
    record.main_materialisation_ms = Number(event.main_stream_materialisation_ms);
  }
}

function premat_partial_hit_progress(record) {
  const total_value = record?.materialisation_ms;
  const wait_value = record?.wait_ms;
  const total_ms = Number(total_value);
  const wait_ms = Number(wait_value);
  if (total_value === null || total_value === undefined
      || wait_value === null || wait_value === undefined
      || !Number.isFinite(total_ms) || total_ms <= 0
      || !Number.isFinite(wait_ms) || wait_ms < 0) {
    return null;
  }
  const raw_percent = 100 * Math.max(0, Math.min(1, (total_ms - wait_ms) / total_ms));
  // A waited hit was observably incomplete at its deadline. Keep its rounded
  // estimate below 100% so it cannot be mistaken for a full hit.
  return Math.max(0, Math.min(95, Math.round(raw_percent / 5) * 5));
}

function premat_build_model(snapshot) {
  const attention_mode = snapshot.attention_mode === "unfused" ? "unfused" : "fused";
  const layers = premat_layer_indices(snapshot);
  const families = premat_families(attention_mode);
  const records = new Map();
  for (const layer_index of layers) {
    for (const family of families) {
      const record = premat_new_record(layer_index, family, attention_mode);
      records.set(record.key, record);
    }
  }

  const frames = [];
  let pending_updates = [];
  const add_zero_update = update => pending_updates.push(update);
  const add_frame = (update, event, frame_state) => {
    frames.push({
      updates: [...pending_updates, update],
      active_layer_index: Number(event.current_layer_index ?? event.layer_index),
      event_sequence: Number(event.sequence),
      frame_state,
      duration_multiplier: update.state === "materialising"
        ? PREMAT_PREMATERIALISING_DURATION_MULTIPLIER
        : 1,
      final: false,
    });
    pending_updates = [];
  };

  const events = [...(snapshot.events || [])]
    .sort((left, right) => Number(left.sequence) - Number(right.sequence));
  for (const event of events) {
    if (!event.family || !Number.isFinite(Number(event.layer_index))) continue;
    const key = premat_candidate_key(event.layer_index, event.family);
    const record = records.get(key);
    if (!record) continue;
    premat_update_from_event(record, event);
    const event_name = String(event.event || "");
    const owner = String(event.owner || "none");
    const state = String(event.new_state || event.state || "");
    const waited = event.outcome === "waited_for_premat"
      || event.reason === "materialising_at_deadline";
    const main_owned = owner === "main" || event.decision === "main_claim";

    // pass_end_release reports the candidate's last state, not a transition.
    // Handle it first so it cannot fabricate another timed state frame.
    if (event.event === "pass_end_release") {
      record.outcome = "INCOMPLETE PASS";
      premat_append_trace(record, "INCOMPLETE PASS");
      continue;
    }

    if (state === "MATERIALISING"
        && (event_name === "materialising" || event_name === "materialising_on_critical_path")) {
      if (main_owned) {
        record.path = "main";
        record.outcome = "COMPLETE MISS";
        premat_append_trace(record, "PREMAT NOT STARTED - MAIN CODE MATERIALISING");
        add_frame({key, state: "main-materialising", outcome: record.outcome}, event, "PREMAT NOT STARTED - MAIN CODE MATERIALISING");
      } else {
        record.path = "premat";
        premat_append_trace(record, "PRE-MATERIALISING");
        add_frame({key, state: "materialising", outcome: record.outcome}, event, "PRE-MATERIALISING");
      }
      continue;
    }

    if (state === "AVAILABLE" && event_name === "available") {
      if (owner === "premat" && !event.critical_path_miss) {
        record.path = "full";
        record.outcome = "FULL HIT";
        premat_append_trace(record, "AVAILABLE");
        add_frame({key, state: "available", outcome: record.outcome}, event, "AVAILABLE");
      }
      continue;
    }

    if (state === "CONSUMING"
        && (event_name === "consuming" || event_name === "critical_path_wait")) {
      if (main_owned) {
        record.path = "main";
        record.outcome = "COMPLETE MISS";
        premat_append_trace(record, "MAIN CODE CONSUMING");
        add_frame({key, state: "main-consuming", outcome: record.outcome}, event, "MAIN CODE CONSUMING");
      } else if (waited || event.critical_path_miss) {
        record.path = "waited";
        record.outcome = "PARTIAL HIT";
        premat_append_trace(record, "WAITING FOR PRE-MATERIALISATION");
        add_frame({key, state: "waiting", outcome: record.outcome}, event, "WAITING FOR PRE-MATERIALISATION");
        premat_append_trace(record, "CONSUMING AFTER WAIT");
        add_frame({key, state: "consuming-waited", outcome: record.outcome}, event, "CONSUMING AFTER WAIT");
      } else {
        record.path = "full";
        record.outcome = "FULL HIT";
        premat_append_trace(record, "CONSUMING - NO WAITING");
        add_frame({key, state: "consuming-full", outcome: record.outcome}, event, "CONSUMING - NO WAITING");
      }
      continue;
    }

    if (state === "CONSUMED" && event_name === "consumed") {
      const terminal_state = record.path === "full"
        ? "consumed-full"
        : record.path === "waited"
          ? "consumed-waited"
          : "consumed-main";
      if (record.path === "none") {
        record.path = "main";
        record.outcome = "COMPLETE MISS";
      }
      if (record.path === "waited") {
        record.completion_at_deadline_percent = premat_partial_hit_progress(record);
      }
      premat_append_trace(record, record.outcome);
      add_zero_update({
        key,
        state: terminal_state,
        outcome: record.outcome,
        completion_at_deadline_percent: record.completion_at_deadline_percent,
      });
      continue;
    }

  }

  frames.push({
    updates: pending_updates,
    active_layer_index: null,
    event_sequence: Number(snapshot.latest_event_sequence ?? 0),
    frame_state: "COMPLETE",
    final: true,
  });
  return {attention_mode, layers, families, records, frames};
}

function premat_state_class(state) {
  const classes = {
    neutral: "premat-neutral",
    pending: "premat-pending",
    materialising: "premat-state-materialising",
    available: "premat-state-available",
    "consuming-full": "premat-state-consuming-full",
    "consumed-full": "premat-state-consumed-full",
    waiting: "premat-state-waiting",
    "consuming-waited": "premat-state-consuming-waited",
    "consumed-waited": "premat-state-consumed-waited",
    "main-materialising": "premat-state-main-materialising",
    "main-consuming": "premat-state-main-consuming",
    "consumed-main": "premat-state-consumed-main",
  };
  return classes[state] || classes.neutral;
}

function premat_row_height(layer_count) {
  const available = Math.min(760, Math.max(360, window.innerHeight * 0.58));
  return Math.max(20, Math.min(42, Math.floor((available - Math.max(0, layer_count - 1) * 3) / Math.max(1, layer_count))));
}

function premat_render_layout(model) {
  const stages = premat_stages(model.attention_mode);
  const row_height = premat_row_height(model.layers.length);
  const first_layer = model.layers[0];
  const size_markup = stages.map(([_label, family]) => {
    if (!family) return '<span class="premat-matrix-size" aria-hidden="true"></span>';
    const record = model.records.get(premat_candidate_key(first_layer, family));
    const size = premat_matrix_size(record?.retained_bytes);
    return `<span class="premat-matrix-size" title="Materialised ${premat_escape(record?.label || family)} matrix size">${premat_escape(size)}</span>`;
  }).join("");
  const size_row = `<div class="premat-matrix-size-row"><div></div><div class="premat-matrix-sizes" style="--premat-stage-count:${stages.length}">${size_markup}</div></div>`;
  const rows = [...model.layers].sort((left, right) => right - left).map(layer_index => {
    const stage_markup = stages.map(([label, family]) => {
      const key = family ? premat_candidate_key(layer_index, family) : "";
      const attributes = family
        ? ` data-premat-key="${premat_escape(key)}" data-premat-family="${premat_escape(family)}"`
        : "";
      const state_class = family ? "premat-pending" : "premat-neutral";
      return `<span class="premat-stage ${state_class}"${attributes} title="${premat_escape(label)}">${premat_escape(label)}</span>`;
    }).join("");
    return `<div class="premat-layer-row" data-premat-layer="${layer_index}"><div class="premat-layer-number">${layer_index + 1}</div><div class="premat-stage-row" style="--premat-stage-count:${stages.length}">${stage_markup}</div></div>`;
  }).join("");
  const container = by_id("premat_layers");
  container.style.setProperty("--premat-row-height", `${row_height}px`);
  container.innerHTML = size_row + rows;
  premat_view.cell_elements = new Map();
  container.querySelectorAll("[data-premat-key]").forEach(element => {
    premat_view.cell_elements.set(element.dataset.prematKey, element);
  });
  premat_view.row_elements = new Map();
  container.querySelectorAll("[data-premat-layer]").forEach(element => {
    premat_view.row_elements.set(Number(element.dataset.prematLayer), element);
  });
  premat_view.active_layer_index = null;
}

function premat_summary_item(label, value) {
  return `<span class="premat-summary-item">${premat_escape(label)}<strong>${premat_escape(value)}</strong></span>`;
}

function premat_render_summary(snapshot, model) {
  const outcomes = {"FULL HIT": 0, "PARTIAL HIT": 0, "COMPLETE MISS": 0};
  for (const record of model.records.values()) outcomes[record.outcome] = (outcomes[record.outcome] || 0) + 1;
  const memory = snapshot.memory || {};
  const margin = Number(memory.device_free_bytes) - Number(memory.global_buffer_bytes);
  const margin_text = Number.isFinite(margin)
    ? `${margin >= 0 ? "+" : "−"}${premat_bytes(Math.abs(margin))}`
    : "—";
  by_id("premat_summary").innerHTML = [
    ["mode", model.attention_mode],
    ["order", snapshot.target_order === "reverse_execution" ? "reverse l+1" : "—"],
    ["priority", snapshot.cuda_stream_priority || "normal"],
    ["buffer margin", margin_text],
    ["headroom", premat_bytes(memory.premat_headroom_bytes)],
    ["full hits", String(outcomes["FULL HIT"])],
    ["partial hits", String(outcomes["PARTIAL HIT"])],
    ["complete misses", String(outcomes["COMPLETE MISS"])],
  ].map(([label, value]) => premat_summary_item(label, value)).join("");
}

function premat_apply_update(update) {
  const element = premat_view.cell_elements.get(update.key);
  if (!element) return;
  element.classList.remove(...PREMAT_STATE_CLASSES);
  element.classList.add(premat_state_class(update.state));
  element.style.removeProperty("--premat-partial-progress");
  const record = premat_view.active_model?.records.get(update.key);
  const progress_value = update.completion_at_deadline_percent;
  const progress = Number(progress_value);
  const has_progress = update.state === "consumed-waited"
    && progress_value !== null
    && progress_value !== undefined
    && Number.isFinite(progress);
  if (has_progress) element.style.setProperty("--premat-partial-progress", `${progress}%`);
  const state_label = String(update.state).replaceAll("-", " ").toUpperCase();
  const progress_label = has_progress ? `; approximately ${progress}% time-progress at deadline` : "";
  element.title = `${record?.label || update.key}; ${state_label}; ${update.outcome || record?.outcome || ""}${progress_label}`;
}

function premat_frame_delay(frame) {
  if (frame?.final) return PREMAT_FINAL_HOLD_MS;
  return premat_view.state_duration_ms * Number(frame?.duration_multiplier || 1);
}

function premat_set_active_layer(layer_index) {
  if (premat_view.active_layer_index !== null) {
    premat_view.row_elements.get(premat_view.active_layer_index)?.classList.remove("premat-active-layer");
  }
  premat_view.active_layer_index = Number.isFinite(layer_index) ? Number(layer_index) : null;
  if (premat_view.active_layer_index !== null) {
    premat_view.row_elements.get(premat_view.active_layer_index)?.classList.add("premat-active-layer");
  }
}

function premat_apply_frame(frame) {
  for (const update of frame.updates) premat_apply_update(update);
  premat_set_active_layer(frame.active_layer_index);
  by_id("premat_update").textContent = frame.final
    ? "complete microstep · final outcomes"
    : `${frame.frame_state} · event ${frame.event_sequence}`;
}

function premat_clear_timer() {
  if (premat_view.playback_timer !== null) clearTimeout(premat_view.playback_timer);
  premat_view.playback_timer = null;
}

function premat_finish_playback() {
  premat_view.playback_running = false;
  if (premat_view.pending_snapshot) {
    const pending = premat_view.pending_snapshot;
    premat_view.pending_snapshot = null;
    premat_start_snapshot(pending);
  }
}

function premat_advance_playback() {
  premat_clear_timer();
  if (!premat_view.playing || !premat_view.playback_running) return;
  if (premat_view.frame_index >= premat_view.frames.length) {
    premat_finish_playback();
    return;
  }
  const frame = premat_view.frames[premat_view.frame_index++];
  premat_apply_frame(frame);
  const delay = premat_frame_delay(frame);
  premat_view.playback_timer = setTimeout(premat_advance_playback, delay);
}

function premat_snapshot_complete(snapshot) {
  if (!snapshot) return false;
  if (snapshot.pass_complete === true) return true;
  const events = snapshot.events || [];
  return events.length > 0 && events[events.length - 1].event === "pass_end";
}

function premat_render_inspector(snapshot, model) {
  const rows = [];
  for (const layer_index of [...model.layers].sort((left, right) => right - left)) {
    for (const family of model.families) {
      const record = model.records.get(premat_candidate_key(layer_index, family));
      const progress_value = record.completion_at_deadline_percent;
      const progress = Number(progress_value);
      const outcome = record.outcome === "PARTIAL HIT"
          && progress_value !== null
          && progress_value !== undefined
          && Number.isFinite(progress)
        ? `PARTIAL HIT · ~${progress}% TIME-PROGRESS`
        : record.outcome;
      rows.push(`<tr><td>${layer_index + 1}</td><td>${premat_escape(record.label)}</td><td>${premat_matrix_size(record.retained_bytes)}</td><td>${premat_escape(record.trace.join(" → ") || "—")}</td><td>${premat_escape(outcome)}</td><td>${premat_ms(record.wait_ms)}</td><td>${premat_ms(record.materialisation_ms)}</td><td>${premat_ms(record.main_materialisation_ms)}</td><td>${premat_escape(record.admission_reason)}</td></tr>`);
    }
  }
  by_id("premat_inspector_step").textContent = String(snapshot.optimizer_update ?? "—");
  by_id("premat_inspector_detail").textContent = `${model.layers.length} layers · ${rows.length} matrix opportunities`;
  by_id("premat_inspector_body").innerHTML = rows.join("");
  by_id("premat_inspect_button").disabled = false;
}

function premat_start_snapshot(snapshot) {
  if (!premat_snapshot_complete(snapshot)) return;
  premat_clear_timer();
  const model = premat_build_model(snapshot);
  premat_view.active_snapshot = snapshot;
  premat_view.active_model = model;
  premat_view.frames = model.frames;
  premat_view.frame_index = 0;
  premat_view.playback_running = true;
  by_id("premat_step").textContent = String(snapshot.optimizer_update ?? "—");
  by_id("premat_mode").textContent = premat_memory_rule(snapshot);
  premat_render_layout(model);
  premat_render_summary(snapshot, model);
  premat_render_inspector(snapshot, model);
  premat_advance_playback();
}

function premat_receive_snapshot(snapshot) {
  if (!premat_snapshot_complete(snapshot)) return;
  const update = Number(snapshot.optimizer_update ?? 0);
  premat_view.latest_received_update = Math.max(premat_view.latest_received_update, update);
  if (premat_view.playback_running) {
    const pending_update = Number(premat_view.pending_snapshot?.optimizer_update ?? -1);
    if (update > Number(premat_view.active_snapshot?.optimizer_update ?? -1) && update >= pending_update) {
      premat_view.pending_snapshot = snapshot;
    }
    return;
  }
  if (!premat_view.playing) {
    premat_view.pending_snapshot = snapshot;
    return;
  }
  premat_start_snapshot(snapshot);
}

function premat_sync_play_button() {
  const button = by_id("premat_play_toggle");
  button.textContent = premat_view.playing ? "Ⅱ" : "▶";
  button.title = premat_view.playing ? "Pause Premat playback" : "Resume Premat playback";
  button.setAttribute("aria-label", button.title);
}

function premat_toggle_playback() {
  premat_view.playing = !premat_view.playing;
  premat_sync_play_button();
  if (!premat_view.playing) {
    premat_clear_timer();
    return;
  }
  if (premat_view.playback_running) {
    premat_advance_playback();
  } else if (premat_view.pending_snapshot) {
    const pending = premat_view.pending_snapshot;
    premat_view.pending_snapshot = null;
    premat_start_snapshot(pending);
  }
}

function premat_reset() {
  premat_clear_timer();
  premat_view.latest_received_update = 0;
  premat_view.pending_snapshot = null;
  premat_view.active_snapshot = null;
  premat_view.active_model = null;
  premat_view.frames = [];
  premat_view.frame_index = 0;
  premat_view.playback_running = false;
  premat_view.cell_elements.clear();
  premat_view.row_elements.clear();
  premat_view.active_layer_index = null;
  by_id("premat_step").textContent = "—";
  by_id("premat_update").textContent = "No complete microstep";
  by_id("premat_summary").innerHTML = "";
  by_id("premat_layers").innerHTML = "";
  by_id("premat_inspect_button").disabled = true;
}

function premat_sync_tab(premat_selected = null) {
  if (premat_selected === null) {
    premat_selected = Boolean(document.querySelector?.('[data-detail-tab="premat"].active'));
  }
  by_id("premat_chart_group").hidden = !premat_selected;
  by_id("depth_chart_group").hidden = premat_selected;
  if (premat_selected && !premat_view.active_snapshot) {
    by_id("premat_mode").textContent = "waiting for a complete captured microstep";
  }
}

window.premat_apply_detail_tab = premat_sync_tab;

async function refresh_premat() {
  const run_id = app.current_run_id;
  if (!run_id) {
    premat_view.run_id = null;
    premat_reset();
    premat_sync_tab();
    return;
  }
  if (premat_view.run_id !== run_id) {
    premat_view.run_id = run_id;
    premat_reset();
    premat_sync_tab();
  }
  const serial = ++premat_view.request_serial;
  try {
    const after = premat_view.latest_received_update;
    const payload = await fetch_json(`/api/premat?run=${encodeURIComponent(run_id)}&after=${after}`);
    if (serial !== premat_view.request_serial || run_id !== app.current_run_id) return;
    if (payload?.latest) premat_receive_snapshot(payload.latest);
  } catch (_error) {
    if (serial === premat_view.request_serial && !premat_view.active_snapshot) {
      by_id("premat_mode").textContent = "Premat telemetry unavailable";
    }
  }
}

window.premat_refresh = refresh_premat;

window.addEventListener("DOMContentLoaded", () => {
  const duration = by_id("premat_state_duration");
  const duration_label = by_id("premat_state_duration_label");
  const update_duration = () => {
    premat_view.state_duration_ms = Math.max(10, Number(duration?.value || 250));
    if (duration_label) duration_label.textContent = `${(premat_view.state_duration_ms / 1000).toFixed(2)} s`;
    if (premat_view.playback_running && premat_view.playing) {
      premat_clear_timer();
      const visible_frame = premat_view.frames[premat_view.frame_index - 1];
      const delay = premat_frame_delay(visible_frame);
      premat_view.playback_timer = setTimeout(premat_advance_playback, delay);
    }
  };
  duration?.addEventListener("input", update_duration);
  by_id("premat_play_toggle")?.addEventListener("click", premat_toggle_playback);
  by_id("premat_inspect_button")?.addEventListener("click", () => {
    const dialog = by_id("premat_inspector");
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
  });
  by_id("premat_inspector_close")?.addEventListener("click", () => by_id("premat_inspector").close());
  update_duration();
  premat_sync_play_button();
  premat_sync_tab();
  refresh_premat();
  setInterval(refresh_premat, 750);
});

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    premat_build_model,
    premat_families,
    premat_family_label,
    premat_matrix_size,
    premat_memory_rule,
    premat_partial_hit_progress,
    premat_render_layout,
    premat_snapshot_complete,
  };
}
// ^^^ THOG
