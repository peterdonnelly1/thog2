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
  active_panel: "recap",
  history_request_serial: 0,
  history_latest_update: 0,
  history_snapshots: [],
  history_model: null,
  inspector_return_panel: "recap",
  inspector_snapshot: null,
  inspector_model: null,
};

function premat_escape(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[character]));
}

function premat_bytes(value) {
  if (value === null || value === undefined) return "—";
  const bytes = Number(value);
  if (!Number.isFinite(bytes)) return "—";
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}

function premat_ms(value) {
  if (value === null || value === undefined) return "—";
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
    ? {QK: "ATTN QK", V: "ATTN V", O: "ATTN O", UP: "MLP UP", DOWN: "MLP DN"}
    : {QKV: "ATTN FUSED · QKV", O: "ATTN O", UP: "MLP UP", DOWN: "MLP DN"};
  return labels[family] || String(family);
}

function premat_stages(attention_mode) {
  return attention_mode === "unfused"
    ? [
        ["LN1", null], ["ATTN QK", "QK"], ["score", null],
        ["scale / mask", null], ["softmax", null], ["ATTN V", "V"],
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

function premat_new_record(layer_index, family, attention_mode, snapshot) {
  const target_offset = Number(snapshot?.target_offset ?? snapshot?.target_layer ?? 1);
  const target_order = String(snapshot?.matrix_order ?? snapshot?.weight_matrix_target_order ?? snapshot?.target_order ?? "r_to_l");
  const ordered_families = target_order === "r_to_l"
    ? [...premat_families(attention_mode)].reverse()
    : premat_families(attention_mode);
  return {
    key: premat_candidate_key(layer_index, family),
    layer_index,
    family,
    label: premat_family_label(family, attention_mode),
    target_offset,
    target_order,
    target_order_position: ordered_families.indexOf(family),
    trace: [],
    outcome: "NOT YET REACHED",
    path: "none",
    admission_reason: "not checked",
    admission_history: [],
    first_considered_ms: null,
    first_observed_admissible_ms: null,
    first_observed_admissible_headroom_bytes: null,
    first_observed_admissible_charged_bytes: null,
    submission_ms: null,
    completion_observed_ms: null,
    deadline_ms: null,
    consumption_ms: null,
    queue_depth: null,
    cumulative_charged_bytes: null,
    submission_queue_depth: null,
    submission_charged_bytes: null,
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
  if (Number.isFinite(Number(event.target_offset))) record.target_offset = Number(event.target_offset);
  if (event.target_order) record.target_order = String(event.target_order);
  if (Number.isFinite(Number(event.target_order_position))) {
    record.target_order_position = Number(event.target_order_position);
  }
  if (Number.isFinite(Number(event.queue_depth))) record.queue_depth = Number(event.queue_depth);
  if (Number.isFinite(Number(event.cumulative_charged_bytes))) {
    record.cumulative_charged_bytes = Number(event.cumulative_charged_bytes);
  }
  const elapsed_ms = Number(event.elapsed_ms);
  if (event.event === "admission_considered" && record.first_considered_ms === null && Number.isFinite(elapsed_ms)) {
    record.first_considered_ms = elapsed_ms;
  }
  if (event.event === "admission_considered" && event.outcome === "first_observed_admissible" && Number.isFinite(elapsed_ms)) {
    record.first_observed_admissible_ms = elapsed_ms;
    record.first_observed_admissible_headroom_bytes = Number.isFinite(Number(event.premat_headroom_bytes)) ? Number(event.premat_headroom_bytes) : null;
    record.first_observed_admissible_charged_bytes = Number.isFinite(Number(event.cumulative_charged_bytes)) ? Number(event.cumulative_charged_bytes) : null;
  }
  if (event.event === "admission_deferred") {
    record.admission_history.push({
      elapsed_ms: Number.isFinite(elapsed_ms) ? elapsed_ms : null,
      reason: String(event.reason || event.admission_reason || "rejected").replaceAll("_", " "),
      headroom_bytes: Number.isFinite(Number(event.premat_headroom_bytes)) ? Number(event.premat_headroom_bytes) : null,
      charged_bytes: Number.isFinite(Number(event.cumulative_charged_bytes)) ? Number(event.cumulative_charged_bytes) : null,
    });
  }
  if (event.event === "materialising" && event.owner === "premat") {
    if (Number.isFinite(elapsed_ms)) record.submission_ms = elapsed_ms;
    record.submission_queue_depth = Number.isFinite(Number(event.queue_depth)) ? Number(event.queue_depth) : null;
    record.submission_charged_bytes = Number.isFinite(Number(event.cumulative_charged_bytes)) ? Number(event.cumulative_charged_bytes) : null;
  }
  if (event.event === "available" && Number.isFinite(elapsed_ms)) record.completion_observed_ms = elapsed_ms;
  if (String(event.event || "").startsWith("deadline_") && Number.isFinite(elapsed_ms)) record.deadline_ms = elapsed_ms;
  if ((event.event === "consuming" || event.event === "critical_path_wait") && Number.isFinite(elapsed_ms)) record.consumption_ms = elapsed_ms;
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
      const record = premat_new_record(layer_index, family, attention_mode, snapshot);
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

function premat_history_model(snapshots) {
  const complete = [...(snapshots || [])]
    .filter(premat_snapshot_complete)
    .sort((left, right) => Number(right.optimizer_update ?? 0) - Number(left.optimizer_update ?? 0));
  const layers = [...new Set(complete.flatMap(premat_layer_indices))]
    .sort((left, right) => left - right);
  const rows = complete.map(snapshot => {
    const model = premat_build_model(snapshot);
    const outcomes = {};
    for (const layer_index of layers) {
      const counts = {full_hits: 0, partial_hits: 0, complete_misses: 0};
      for (const family of model.families) {
        const outcome = model.records.get(premat_candidate_key(layer_index, family))?.outcome;
        if (outcome === "FULL HIT") counts.full_hits += 1;
        else if (outcome === "PARTIAL HIT") counts.partial_hits += 1;
        else if (outcome === "COMPLETE MISS") counts.complete_misses += 1;
      }
      outcomes[layer_index] = counts;
    }
    const memory = snapshot.memory || {};
    const device_free = Number(memory.device_free_bytes);
    const global_buffer = Number(memory.global_buffer_bytes);
    const buffer_margin_bytes = Number.isFinite(device_free) && Number.isFinite(global_buffer)
      ? device_free - global_buffer
      : null;
    const headroom = Number(memory.premat_headroom_bytes);
    return {
      optimizer_update: Number(snapshot.optimizer_update ?? 0),
      buffer_margin_bytes,
      headroom_bytes: Number.isFinite(headroom) ? headroom : null,
      outcomes,
    };
  });
  return {layers, rows};
}

function premat_history_csv(model) {
  const quote = value => `"${String(value ?? "").replace(/"/g, '""')}"`;
  const header = ["step", "buffer_margin_bytes", "headroom_bytes"];
  for (const layer_index of model.layers) {
    const layer = layer_index + 1;
    header.push(
      `layer_${layer}_full_hits`,
      `layer_${layer}_partial_hits`,
      `layer_${layer}_complete_misses`,
    );
  }
  const lines = [header.map(quote).join(",")];
  for (const row of model.rows) {
    const values = [row.optimizer_update, row.buffer_margin_bytes, row.headroom_bytes];
    for (const layer_index of model.layers) {
      const counts = row.outcomes[layer_index] || {};
      values.push(counts.full_hits ?? 0, counts.partial_hits ?? 0, counts.complete_misses ?? 0);
    }
    lines.push(values.map(value => value === null || value === undefined ? "" : String(value)).join(","));
  }
  return lines.join("\r\n") + "\r\n";
}

function premat_csv_cell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function premat_csv_value(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") {
    const encoded = JSON.stringify(value);
    return encoded === undefined ? "" : encoded;
  }
  return value;
}

function premat_optional_number(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function premat_one_indexed(value) {
  const number = premat_optional_number(value);
  return number === null ? null : number + 1;
}

function premat_flatten_event_value(row, prefix, value) {
  if (
    value
    && typeof value === "object"
    && !Array.isArray(value)
  ) {
    const entries = Object.entries(value);
    if (entries.length) {
      for (const [key, nested] of entries) {
        premat_flatten_event_value(row, `${prefix}_${key}`, nested);
      }
      return;
    }
  }
  row[prefix] = value;
}

function premat_raw_event_rows(snapshots) {
  const rows = [];
  const complete = [...(snapshots || [])]
    .filter(premat_snapshot_complete)
    .sort((left, right) => Number(right.optimizer_update ?? 0) - Number(left.optimizer_update ?? 0));
  for (const snapshot of complete) {
    const snapshot_memory = snapshot.memory || {};
    const target_offset = snapshot.target_offset ?? snapshot.target_layer ?? null;
    const matrix_order = snapshot.matrix_order
      ?? snapshot.weight_matrix_target_order
      ?? snapshot.target_order
      ?? "";
    const snapshot_events = [...(snapshot.events || [])]
      .sort((left, right) => Number(left.sequence ?? 0) - Number(right.sequence ?? 0));
    for (const event of snapshot_events) {
      const process_allocated = premat_optional_number(event.process_allocated_bytes);
      const process_reserved = premat_optional_number(event.process_reserved_bytes);
      const device_free = premat_optional_number(event.device_free_bytes);
      const global_buffer = premat_optional_number(
        event.global_buffer_bytes ?? snapshot_memory.global_buffer_bytes,
      );
      const reusable_allocator = premat_optional_number(event.reusable_allocator_bytes);
      const device_margin = premat_optional_number(event.device_free_minus_buffer_bytes);
      const row = {
        step: Number(snapshot.optimizer_update ?? 0),
        snapshot_pass_sequence: snapshot.pass_sequence ?? null,
        event_sequence: event.sequence ?? null,
        event: event.event ?? "",
        event_type: event.event_type ?? event.event ?? "",
        host_time_ns: event.host_time_ns ?? null,
        elapsed_ms: event.elapsed_ms ?? null,
        layer: premat_one_indexed(event.layer_index),
        layer_index: event.layer_index ?? null,
        family: event.family ?? "",
        current_layer: premat_one_indexed(event.current_layer_index),
        current_layer_index: event.current_layer_index ?? null,
        next_layer: premat_one_indexed(event.next_layer_index),
        next_layer_index: event.next_layer_index ?? null,
        target_layer: premat_one_indexed(event.target_layer_index),
        target_layer_index: event.target_layer_index ?? null,
        target_offset: event.target_offset ?? target_offset,
        matrix_order: event.target_order ?? matrix_order,
        matrix_order_position: premat_one_indexed(event.target_order_position),
        attention_mode: snapshot.attention_mode ?? "",
        cuda_stream_priority: snapshot.cuda_stream_priority ?? "",
        headroom_mode: snapshot.headroom_mode ?? event.headroom_policy ?? "",
        diagnostic_layer_delay_ms: snapshot.diagnostic_layer_delay_ms ?? null,
        decision: event.decision ?? "",
        outcome: event.outcome ?? event.final_outcome ?? "",
        final_outcome: event.final_outcome ?? "",
        reason: event.reason ?? event.admission_reason ?? "",
        admission_reason: event.admission_reason ?? "",
        owner: event.owner ?? "",
        old_state: event.old_state ?? "",
        new_state: event.new_state ?? "",
        state: event.state ?? "",
        critical_path_miss: event.critical_path_miss ?? null,
        queue_depth: event.queue_depth ?? null,
        cumulative_charged_bytes: event.cumulative_charged_bytes ?? null,
        charged_retained_bytes: event.charged_retained_bytes ?? null,
        charged_transient_bytes: event.charged_transient_bytes ?? null,
        process_allocated_bytes: process_allocated,
        process_reserved_bytes: process_reserved,
        reusable_allocator_bytes: reusable_allocator ?? (
          process_allocated !== null && process_reserved !== null
            ? Math.max(0, process_reserved - process_allocated)
            : null
        ),
        process_ordinary_peak_bytes: event.process_ordinary_peak_bytes ?? null,
        device_free_bytes: device_free,
        device_used_bytes: event.device_used_bytes ?? null,
        device_total_bytes: event.device_total_bytes ?? null,
        global_buffer_bytes: global_buffer,
        device_ceiling_bytes: event.device_ceiling_bytes ?? null,
        device_free_minus_buffer_bytes: device_margin ?? (
          device_free !== null && global_buffer !== null
            ? device_free - global_buffer
            : null
        ),
        process_headroom_bytes: event.process_headroom_bytes ?? null,
        device_headroom_bytes: event.device_headroom_bytes ?? null,
        premat_headroom_bytes: event.premat_headroom_bytes ?? null,
        predicted_retained_bytes: event.predicted_retained_bytes ?? null,
        predicted_materialisation_peak_bytes: event.predicted_materialisation_peak_bytes ?? null,
        predicted_foreground_overlap_bytes: event.predicted_foreground_overlap_bytes ?? null,
        predicted_envelope_bytes: event.predicted_envelope_bytes ?? null,
        first_considered_ns: event.first_considered_ns ?? null,
        first_observed_admissible_ns: event.first_observed_admissible_ns ?? null,
        submission_ns: event.submission_ns ?? null,
        cuda_completion_observed_ns: event.cuda_completion_observed_ns ?? null,
        deadline_ns: event.deadline_ns ?? null,
        consumption_ns: event.consumption_ns ?? null,
        observed_admission_lag_ms: event.observed_admission_lag_ms ?? null,
        materialisation_ms: event.materialisation_ms ?? null,
        main_stream_materialisation_ms: event.main_stream_materialisation_ms ?? null,
        wait_ms: event.wait_ms ?? null,
      };
      for (const [key, value] of Object.entries(event)) {
        if (key === "detail" || Object.prototype.hasOwnProperty.call(row, key)) continue;
        row[`event_${key}`] = value;
      }
      if (event.detail && typeof event.detail === "object") {
        for (const [key, value] of Object.entries(event.detail)) {
          premat_flatten_event_value(row, `detail_${key}`, value);
        }
      }
      row.raw_event_json = JSON.stringify(event);
      rows.push(row);
    }
  }
  return rows;
}

function premat_raw_event_history_csv(snapshots) {
  const rows = premat_raw_event_rows(snapshots);
  const preferred = [
    "step", "snapshot_pass_sequence", "event_sequence", "event", "event_type",
    "host_time_ns", "elapsed_ms", "layer", "layer_index", "family",
    "current_layer", "current_layer_index", "next_layer", "next_layer_index",
    "target_layer", "target_layer_index", "target_offset", "matrix_order",
    "matrix_order_position", "attention_mode", "cuda_stream_priority",
    "headroom_mode", "diagnostic_layer_delay_ms", "decision", "outcome",
    "final_outcome", "reason", "admission_reason", "owner", "old_state",
    "new_state", "state", "critical_path_miss", "queue_depth",
    "cumulative_charged_bytes", "charged_retained_bytes",
    "charged_transient_bytes", "process_allocated_bytes",
    "process_reserved_bytes", "reusable_allocator_bytes",
    "process_ordinary_peak_bytes", "device_free_bytes", "device_used_bytes",
    "device_total_bytes", "global_buffer_bytes", "device_ceiling_bytes",
    "device_free_minus_buffer_bytes", "process_headroom_bytes",
    "device_headroom_bytes", "premat_headroom_bytes", "predicted_retained_bytes",
    "predicted_materialisation_peak_bytes", "predicted_foreground_overlap_bytes",
    "predicted_envelope_bytes", "first_considered_ns",
    "first_observed_admissible_ns", "submission_ns",
    "cuda_completion_observed_ns", "deadline_ns", "consumption_ns",
    "observed_admission_lag_ms", "materialisation_ms",
    "main_stream_materialisation_ms", "wait_ms",
    "detail_invocation", "detail_trigger", "detail_return_reason",
    "detail_submitted_count", "detail_submitted",
    "detail_blocking_candidate_layer_index",
    "detail_blocking_candidate_family",
    "detail_blocking_candidate_order_position", "detail_blocking_reason",
    "detail_admitted", "detail_process_guard_passed",
    "detail_device_guard_passed", "detail_predicted_process_bytes",
    "detail_predicted_device_used_bytes", "detail_device_ceiling_bytes",
    "detail_predicted_physical_growth_bytes", "detail_process_headroom_bytes",
    "detail_device_headroom_bytes",
    "detail_raw_memory_process_allocated_bytes",
    "detail_raw_memory_process_reserved_bytes",
    "detail_raw_memory_reusable_allocator_bytes",
    "detail_raw_memory_process_ordinary_peak_bytes",
    "detail_raw_memory_device_free_bytes",
    "detail_raw_memory_device_used_bytes",
    "detail_raw_memory_device_total_bytes",
    "detail_raw_memory_device_free_minus_buffer_bytes",
    "detail_charged_memory_process_allocated_bytes",
    "detail_charged_memory_process_reserved_bytes",
    "detail_charged_memory_reusable_allocator_bytes",
    "detail_charged_memory_process_ordinary_peak_bytes",
    "detail_charged_memory_device_free_bytes",
    "detail_charged_memory_device_used_bytes",
    "detail_charged_memory_device_total_bytes",
    "detail_charged_memory_device_free_minus_buffer_bytes",
  ];
  const preferred_set = new Set(preferred);
  const extras = new Set();
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!preferred_set.has(key) && key !== "raw_event_json") extras.add(key);
    }
  }
  const header = [...preferred, ...[...extras].sort(), "raw_event_json"];
  const lines = [header.map(premat_csv_cell).join(",")];
  for (const row of rows) {
    lines.push(
      header.map(key => premat_csv_cell(premat_csv_value(row[key]))).join(","),
    );
  }
  return lines.join("\r\n") + "\r\n";
}

function premat_inspector_rows(snapshot, model) {
  const rows = [];
  const memory = snapshot.memory || {};
  const device_free = Number(memory.device_free_bytes);
  const global_buffer = Number(memory.global_buffer_bytes);
  const buffer_margin_bytes = Number.isFinite(device_free) && Number.isFinite(global_buffer)
    ? device_free - global_buffer
    : null;
  const headroom = Number(memory.premat_headroom_bytes);
  const headroom_bytes = Number.isFinite(headroom) ? headroom : null;
  for (const layer_index of [...model.layers].sort((left, right) => right - left)) {
    for (const family of model.families) {
      const record = model.records.get(premat_candidate_key(layer_index, family));
      const progress_value = record.completion_at_deadline_percent;
      const progress = Number(progress_value);
      const outcome_text = record.outcome === "PARTIAL HIT"
          && progress_value !== null
          && progress_value !== undefined
          && Number.isFinite(progress)
        ? `PARTIAL HIT · ~${progress}% TIME-PROGRESS`
        : record.outcome;
      const target_text = `l+${record.target_offset} · ${record.target_order} #${record.target_order_position + 1}`;
      const admission_history = record.admission_history.length
        ? record.admission_history.map(item => `${premat_ms(item.elapsed_ms)} ${item.reason}; headroom ${premat_bytes(item.headroom_bytes)}; charged ${premat_bytes(item.charged_bytes)}`).join(" | ")
        : "—";
      const first_observed_text = record.first_observed_admissible_ms === null
        ? "—"
        : `${premat_ms(record.first_observed_admissible_ms)}; headroom ${premat_bytes(record.first_observed_admissible_headroom_bytes)}; prior charged ${premat_bytes(record.first_observed_admissible_charged_bytes)}`;
      const processing = record.trace.length > 1
        ? record.trace[record.trace.length - 2]
        : (record.trace[0] || "—");
      rows.push({
        step: Number(snapshot.optimizer_update ?? 0),
        layer: layer_index + 1,
        matrix: record.label,
        target_text,
        target_offset: record.target_offset,
        target_order: record.target_order,
        target_order_position: record.target_order_position + 1,
        retained_bytes: record.retained_bytes,
        buffer_margin_bytes,
        headroom_bytes,
        first_considered_ms: record.first_considered_ms,
        first_observed_admissible_ms: record.first_observed_admissible_ms,
        first_observed_admissible_headroom_bytes: record.first_observed_admissible_headroom_bytes,
        first_observed_admissible_charged_bytes: record.first_observed_admissible_charged_bytes,
        first_observed_text,
        admission_history,
        submission_ms: record.submission_ms,
        completion_observed_ms: record.completion_observed_ms,
        deadline_ms: record.deadline_ms,
        consumption_ms: record.consumption_ms,
        processing,
        processing_trace: record.trace.join(" -> "),
        outcome: record.outcome,
        outcome_text,
        progress_percent: record.completion_at_deadline_percent,
        submission_queue_depth: record.submission_queue_depth,
        submission_charged_bytes: record.submission_charged_bytes,
        wait_ms: record.wait_ms,
        materialisation_ms: record.materialisation_ms,
        main_materialisation_ms: record.main_materialisation_ms,
      });
    }
  }
  return rows;
}

function premat_detailed_history_csv(snapshots) {
  const header = [
    "step", "layer", "matrix", "target", "target_offset", "matrix_order",
    "matrix_order_position", "size_bytes", "buffer_margin_bytes", "headroom_bytes",
    "first_considered_ms", "first_observed_admissible_ms",
    "first_observed_admissible_headroom_bytes",
    "first_observed_admissible_prior_charged_bytes", "admission_history",
    "submission_ms", "completion_observed_ms", "deadline_ms", "consumption_ms",
    "processing", "processing_trace", "outcome", "progress_percent",
    "submission_queue_depth", "submission_charged_bytes", "wait_ms",
    "premat_materialisation_ms", "main_materialisation_ms",
  ];
  const complete = [...(snapshots || [])]
    .filter(premat_snapshot_complete)
    .sort((left, right) => Number(right.optimizer_update ?? 0) - Number(left.optimizer_update ?? 0));
  const lines = [header.map(premat_csv_cell).join(",")];
  for (const snapshot of complete) {
    const model = premat_build_model(snapshot);
    for (const row of premat_inspector_rows(snapshot, model)) {
      lines.push([
        row.step, row.layer, row.matrix, row.target_text, row.target_offset,
        row.target_order, row.target_order_position, row.retained_bytes,
        row.buffer_margin_bytes, row.headroom_bytes, row.first_considered_ms,
        row.first_observed_admissible_ms,
        row.first_observed_admissible_headroom_bytes,
        row.first_observed_admissible_charged_bytes, row.admission_history,
        row.submission_ms, row.completion_observed_ms, row.deadline_ms,
        row.consumption_ms, row.processing, row.processing_trace, row.outcome,
        row.progress_percent, row.submission_queue_depth,
        row.submission_charged_bytes, row.wait_ms, row.materialisation_ms,
        row.main_materialisation_ms,
      ].map(premat_csv_cell).join(","));
    }
  }
  return lines.join("\r\n") + "\r\n";
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
    ["target", `l+${Number(snapshot.target_offset ?? snapshot.target_layer ?? 1)}`],
    ["matrix order", String(snapshot.matrix_order ?? snapshot.weight_matrix_target_order ?? snapshot.target_order ?? "r_to_l")],
    ["priority", snapshot.cuda_stream_priority || "normal"],
    ["layer delay", `${Number(snapshot.diagnostic_layer_delay_ms || 0)} ms`],
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
  const rows = premat_inspector_rows(snapshot, model);
  const markup = rows.map(row => {
    const submitted_completed = `${premat_ms(row.submission_ms)} / ${premat_ms(row.completion_observed_ms)}`;
    const progress_text = row.progress_percent === null ? "—" : `~${row.progress_percent}%`;
    const queue_charge = `${row.submission_queue_depth ?? "—"} / ${premat_bytes(row.submission_charged_bytes)}`;
    return `<tr><td>${row.layer}</td><td>${premat_escape(row.matrix)}</td><td>${premat_escape(row.target_text)}</td><td>${premat_matrix_size(row.retained_bytes)}</td><td>${premat_ms(row.first_considered_ms)}</td><td>${row.first_observed_text}</td><td>${premat_escape(row.admission_history)}</td><td>${submitted_completed}</td><td>${premat_ms(row.deadline_ms)}</td><td>${premat_ms(row.consumption_ms)}</td><td>${premat_escape(row.processing)}</td><td>${premat_escape(row.outcome_text)}</td><td>${premat_escape(progress_text)}</td><td>${premat_escape(queue_charge)}</td><td>${premat_ms(row.wait_ms)}</td><td>${premat_ms(row.materialisation_ms)}</td><td>${premat_ms(row.main_materialisation_ms)}</td></tr>`;
  });
  premat_view.inspector_snapshot = snapshot;
  premat_view.inspector_model = model;
  by_id("premat_inspector_step").textContent = String(snapshot.optimizer_update ?? "—");
  by_id("premat_inspector_detail").textContent = `${model.layers.length} layers · ${rows.length} matrix opportunities`;
  by_id("premat_inspector_body").innerHTML = markup.join("");
  by_id("premat_inspector_download").disabled = rows.length === 0;
  by_id("premat_inspect_button").disabled = false;
}
function premat_show_panel(panel_name) {
  const panel = ["inspector", "history"].includes(panel_name) ? panel_name : "recap";
  premat_view.active_panel = panel;
  by_id("premat_recap_view").hidden = panel !== "recap";
  by_id("premat_inspector").hidden = panel !== "inspector";
  by_id("premat_history").hidden = panel !== "history";
  by_id("premat_inspect_button")?.setAttribute("aria-pressed", String(panel === "inspector"));
  by_id("premat_history_button")?.setAttribute("aria-pressed", String(panel === "history"));
}

function premat_find_history_snapshot(step) {
  const update = Number(step);
  return premat_view.history_snapshots.find(
    snapshot => Number(snapshot.optimizer_update ?? 0) === update,
  ) || null;
}

function premat_open_history_step(step) {
  const snapshot = premat_find_history_snapshot(step);
  if (!snapshot) return;
  premat_view.inspector_return_panel = "history";
  premat_render_inspector(snapshot, premat_build_model(snapshot));
  premat_show_panel("inspector");
}

function premat_close_inspector() {
  const return_panel = premat_view.inspector_return_panel === "history" ? "history" : "recap";
  premat_view.inspector_return_panel = "recap";
  premat_show_panel(return_panel);
}

function premat_render_history(snapshots) {
  const model = premat_history_model(snapshots);
  premat_view.history_model = model;
  const header = [
    "<tr><th>Step</th><th>Buffer margin</th><th>Headroom</th>",
    ...model.layers.map(layer_index => `<th>Layer ${layer_index + 1}<br>F / P / M</th>`),
    "<th>Download</th></tr>",
  ].join("");
  const download_icon = '<svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"></path><path d="m7 10 5 5 5-5"></path><path d="M5 21h14"></path></svg>';
  const rows = model.rows.map(row => {
    const step = Number(row.optimizer_update);
    const layer_cells = model.layers.map(layer_index => {
      const counts = row.outcomes[layer_index] || {};
      const text = `F ${counts.full_hits ?? 0} · P ${counts.partial_hits ?? 0} · M ${counts.complete_misses ?? 0}`;
      return `<td class="premat-history-outcomes" title="Layer ${layer_index + 1}: ${premat_escape(text)}">${premat_escape(text)}</td>`;
    }).join("");
    const download = `<td class="premat-history-download-cell"><button class="premat-history-download-button" type="button" data-premat-history-download="${step}" aria-label="Download detailed Premat data for step ${step}" title="Download step ${step}">${download_icon}</button></td>`;
    return `<tr class="premat-history-row" data-premat-history-step="${step}" tabindex="0" role="button" aria-label="Open detailed Premat data for step ${step}"><td><strong>${step}</strong></td><td>${premat_escape(premat_bytes(row.buffer_margin_bytes))}</td><td>${premat_escape(premat_bytes(row.headroom_bytes))}</td>${layer_cells}${download}</tr>`;
  });
  by_id("premat_history_head").innerHTML = header;
  const body = by_id("premat_history_body");
  body.innerHTML = rows.length
    ? rows.join("")
    : `<tr><td class="premat-history-empty" colspan="${Math.max(4, model.layers.length + 4)}">No retained Premat history for this run.</td></tr>`;
  body.querySelectorAll("[data-premat-history-step]").forEach(element => {
    const open = () => premat_open_history_step(element.dataset.prematHistoryStep);
    element.addEventListener("click", event => {
      if (event.target.closest("[data-premat-history-download]")) return;
      open();
    });
    element.addEventListener("keydown", event => {
      if (event.target.closest("[data-premat-history-download]")) return;
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });
  });
  body.querySelectorAll("[data-premat-history-download]").forEach(button => {
    button.addEventListener("click", event => {
      event.stopPropagation();
      const snapshot = premat_find_history_snapshot(button.dataset.prematHistoryDownload);
      if (snapshot) premat_download_step(snapshot);
    });
  });
  by_id("premat_history_detail").textContent = `${model.rows.length} retained steps · ${model.layers.length} layers · select a row for matrix detail`;
  by_id("premat_history_download").disabled = model.rows.length === 0;
  by_id("premat_history_raw_download").disabled = model.rows.length === 0;
}
function premat_artifact_name() {
  const run_id = String(app.current_run_id || "");
  const run = (app.runs || []).find(candidate => {
    const identifier = typeof run_identifier === "function"
      ? run_identifier(candidate)
      : (candidate.dashboard_run_id ?? candidate.local_run_id ?? candidate.id);
    return String(identifier ?? "") === run_id;
  });
  return String(run?.artifact_name || run?.run_name || run_id || "premat");
}

function premat_download_csv(csv, filename) {
  const blob = new Blob(["\ufeff", csv], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function premat_artifact_file_stem() {
  return premat_artifact_name().replace(/[^A-Za-z0-9._-]+/g, "_") || "premat";
}

function premat_download_step(snapshot) {
  if (!premat_snapshot_complete(snapshot)) return;
  const step = Number(snapshot.optimizer_update ?? 0);
  premat_download_csv(
    premat_detailed_history_csv([snapshot]),
    `${premat_artifact_file_stem()}_step_${step}.csv`,
  );
}

function premat_download_visible_step() {
  if (premat_view.inspector_snapshot) premat_download_step(premat_view.inspector_snapshot);
}

function premat_download_history() {
  const snapshots = premat_view.history_snapshots.filter(premat_snapshot_complete);
  if (!snapshots.length) return;
  premat_download_csv(
    premat_detailed_history_csv(snapshots),
    `${premat_artifact_file_stem()}.csv`,
  );
}

function premat_download_raw_event_history() {
  const snapshots = premat_view.history_snapshots.filter(premat_snapshot_complete);
  if (!snapshots.length) return;
  premat_download_csv(
    premat_raw_event_history_csv(snapshots),
    `${premat_artifact_file_stem()}_premat_raw_events.csv`,
  );
}

async function premat_load_history(force = false) {
  const run_id = app.current_run_id;
  if (!run_id || premat_view.active_panel !== "history") return;
  if (!force && premat_view.history_model && premat_view.history_latest_update >= premat_view.latest_received_update) {
    premat_render_history(premat_view.history_snapshots);
    return;
  }
  const serial = ++premat_view.history_request_serial;
  by_id("premat_history_detail").textContent = "Loading retained steps…";
  try {
    const payload = await fetch_json(`/api/premat?run=${encodeURIComponent(run_id)}&history=1`);
    if (serial !== premat_view.history_request_serial || run_id !== app.current_run_id) return;
    premat_view.history_snapshots = Array.isArray(payload?.history) ? payload.history : [];
    premat_view.history_latest_update = Number(payload?.latest_update ?? 0);
    premat_render_history(premat_view.history_snapshots);
  } catch (_error) {
    if (serial !== premat_view.history_request_serial) return;
    by_id("premat_history_detail").textContent = "Premat history unavailable";
    by_id("premat_history_download").disabled = true;
    by_id("premat_history_raw_download").disabled = true;
  }
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
  if (!(premat_view.active_panel === "inspector" && premat_view.inspector_return_panel === "history")) {
    premat_render_inspector(snapshot, model);
  }
  by_id("premat_history_button").disabled = false;
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
  premat_view.history_request_serial += 1;
  premat_view.history_latest_update = 0;
  premat_view.history_snapshots = [];
  premat_view.history_model = null;
  premat_view.inspector_return_panel = "recap";
  premat_view.inspector_snapshot = null;
  premat_view.inspector_model = null;
  by_id("premat_step").textContent = "—";
  by_id("premat_update").textContent = "No complete microstep";
  by_id("premat_summary").innerHTML = "";
  by_id("premat_layers").innerHTML = "";
  by_id("premat_history_head").innerHTML = "";
  by_id("premat_history_body").innerHTML = "";
  by_id("premat_history_detail").textContent = "Loading retained steps…";
  by_id("premat_inspect_button").disabled = true;
  by_id("premat_history_button").disabled = true;
  by_id("premat_history_download").disabled = true;
  by_id("premat_history_raw_download").disabled = true;
  by_id("premat_inspector_download").disabled = true;
  premat_show_panel("recap");
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
    if (payload?.latest) {
      premat_receive_snapshot(payload.latest);
      if (premat_view.active_panel === "history") void premat_load_history();
    }
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
    if (!premat_view.active_snapshot || !premat_view.active_model) return;
    premat_view.inspector_return_panel = "recap";
    premat_render_inspector(premat_view.active_snapshot, premat_view.active_model);
    premat_show_panel("inspector");
  });
  by_id("premat_inspector_close")?.addEventListener("click", premat_close_inspector);
  by_id("premat_inspector_download")?.addEventListener("click", premat_download_visible_step);
  by_id("premat_history_button")?.addEventListener("click", () => {
    premat_show_panel("history");
    void premat_load_history(true);
  });
  by_id("premat_history_close")?.addEventListener("click", () => premat_show_panel("recap"));
  by_id("premat_history_download")?.addEventListener("click", premat_download_history);
  by_id("premat_history_raw_download")?.addEventListener("click", premat_download_raw_event_history);
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
    premat_detailed_history_csv,
    premat_history_csv,
    premat_raw_event_history_csv,
    premat_raw_event_rows,
    premat_inspector_rows,
    premat_history_model,
    premat_matrix_size,
    premat_memory_rule,
    premat_partial_hit_progress,
    premat_render_layout,
    premat_snapshot_complete,
  };
}
// ^^^ THOG

