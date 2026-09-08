// vvv THOG dedicated Premat polling and fused/unfused lifecycle rendering
"use strict";

const premat_view = {
  run_id: null,
  request_serial: 0,
  sort_key: "sequence",
  sort_descending: true,
  latest_payload: null,
  newest_frame_key: null,
  frame_queue: [],
  playback_rate: 1,
  playback_timer: null,
};

const PREMAT_PLAYBACK_BASE_MS = 750;
const PREMAT_PLAYBACK_QUEUE_LIMIT = 128;

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

function premat_candidate_map(snapshot) {
  const result = new Map();
  for (const candidate of snapshot?.candidates || []) {
    result.set(`${candidate.layer_index}:${candidate.family}`, candidate);
  }
  return result;
}

function premat_stage_markup(layer_index, label, family, candidates) {
  const candidate = family ? candidates.get(`${layer_index}:${family}`) : null;
  const state = candidate?.state || null;
  const classes = ["premat-stage", family ? (state || "UNAVAILABLE").toLowerCase() : "premat-neutral"];
  if (candidate?.critical_path_miss) classes.push("critical-path-miss");
  const detail = candidate
    ? `${state}; owner=${candidate.owner}; admission=${candidate.admission_reason}`
    : "execution stage; no materialised candidate";
  return `<span class="${classes.join(" ")}" title="${premat_escape(detail)}">${premat_escape(label)}</span>`;
}

function premat_layer_markup(layer_index, role, snapshot, candidates) {
  const attention = snapshot.attention_mode === "unfused"
    ? [["ATTN UNFUSED · QK","QK"],["score",null],["scale / mask",null],["softmax",null],["ATTN UNFUSED · V","V"],["attention",null],["O","O"]]
    : [["ATTN FUSED · QKV","QKV"],["attention",null],["O","O"]];
  const stages = [["LN1",null], ...attention, ["resid",null], ["LN2",null], ["MLP UP","UP"], ["GELU",null], ["MLP DN","DOWN"], ["resid",null]];
  const resolved_layer = layer_index === null || layer_index === undefined ? null : layer_index;
  const layer_label = resolved_layer === null ? "none (final layer)" : `layer ${resolved_layer}`;
  const step_label = `step ${snapshot.optimizer_update ?? "—"}`;
  const stage_markup = stages
    .map(([label, family]) => premat_stage_markup(resolved_layer, label, family, candidates))
    .join('<span class="premat-stage-arrow" aria-hidden="true">→</span>');
  return `<article class="premat-layer"><header><strong>${step_label} · ${role} · ${layer_label}</strong></header><div class="premat-stage-row">${stage_markup}</div></article>`;
}

function premat_pair(value, other) {
  return `${premat_bytes(value)} / ${premat_bytes(other)}`;
}

function premat_sort_value(event, key) {
  const value = event?.[key];
  const numeric = Number(value);
  return value !== null && value !== "" && Number.isFinite(numeric)
    ? numeric
    : String(value ?? "").toLowerCase();
}

function premat_visible_events(snapshot) {
  const query = by_id("premat_event_filter")?.value.trim().toLowerCase() || "";
  const events = (snapshot?.events || []).filter(event =>
    !query || JSON.stringify(event).toLowerCase().includes(query)
  );
  events.sort((left, right) => {
    const left_value = premat_sort_value(left, premat_view.sort_key);
    const right_value = premat_sort_value(right, premat_view.sort_key);
    const comparison = typeof left_value === "number" && typeof right_value === "number"
      ? left_value - right_value
      : String(left_value).localeCompare(String(right_value));
    return premat_view.sort_descending ? -comparison : comparison;
  });
  return events.slice(0, 256);
}

function premat_sync_tab(premat_selected = null) {
  if (premat_selected === null) {
    premat_selected = Boolean(document.querySelector?.('[data-detail-tab="premat"].active'));
  }
  const snapshot = premat_view.latest_payload?.latest;
  by_id("premat_chart_group").hidden = !premat_selected;
  by_id("depth_chart_group").hidden = premat_selected;
  if (premat_selected && !snapshot) {
    by_id("premat_mode").textContent = "waiting for telemetry";
    by_id("premat_update").textContent = "No Premat snapshot for this run";
  }
}

window.premat_apply_detail_tab = premat_sync_tab;

function render_premat(payload) {
  premat_view.latest_payload = payload;
  const snapshot = payload?.latest;
  premat_sync_tab();
  if (!snapshot) return;
  by_id("premat_mode").textContent = `${snapshot.attention_mode} · ${String(snapshot.headroom_mode || "").replaceAll("_", " ")}`;
  const shown_events = (snapshot.events || []).length;
  const total_events = Number(snapshot.event_count ?? shown_events);
  by_id("premat_update").textContent = `optimizer update ${snapshot.optimizer_update ?? "—"} · event ${snapshot.latest_event_sequence ?? total_events}`;
  const memory = snapshot.memory || {};
  const memory_items = [
    ["Process allocated", memory.process_allocated_bytes],
    ["Process reserved", memory.process_reserved_bytes],
    ["Process ordinary peak", memory.process_ordinary_peak_bytes],
    ["Device free", memory.device_free_bytes],
    ["Device used", memory.device_used_bytes],
    [`Active ceiling · ${String(memory.active_ceiling_scope || "unknown").replaceAll("_", " ")}`, memory.active_ceiling_bytes],
    ["Premat admission buffer", memory.global_buffer_bytes],
    ["Buffer margin now", (
      Number(memory.device_free_bytes) >= Number(memory.global_buffer_bytes)
        ? `met · +${premat_bytes(Number(memory.device_free_bytes) - Number(memory.global_buffer_bytes))}`
        : `BREACHED · −${premat_bytes(Number(memory.global_buffer_bytes) - Number(memory.device_free_bytes))}`
    )],
    ["Premat headroom", memory.premat_headroom_bytes],
    ["Premat retained", memory.premat_retained_bytes],
  ];
  const aggregate = snapshot.aggregate || {};
  memory_items.push(
    ["Premat admitted", String(aggregate.admitted ?? 0)],
    ["Premat hits", `${aggregate.fully_hidden_hits ?? 0} hidden · ${aggregate.waited_hits ?? 0} waited`],
    ["Main fallbacks", String(aggregate.ordinary_deadline_materialisations ?? 0)],
  );
  const queue_head = snapshot.queue_head || memory.next_candidate;
  const queue_label = queue_head
    ? `layer ${queue_head.layer_index} ${queue_head.family}${queue_head.deferred ? ` · defer: ${queue_head.admission_reason}` : ""}`
    : "none";
  memory_items.push(["Next candidate", queue_label]);
  by_id("premat_memory").innerHTML = memory_items.map(([label, value]) => `<div class="premat-memory-item"><span>${premat_escape(label)}</span><strong>${typeof value === "number" ? premat_bytes(value) : premat_escape(value)}</strong></div>`).join("");
  const candidates = premat_candidate_map(snapshot);
  by_id("premat_layers").innerHTML =
    premat_layer_markup(snapshot.next_layer_index, "lookahead", snapshot, candidates)
    + premat_layer_markup(snapshot.current_layer_index, "current", snapshot, candidates);
  const events = premat_visible_events(snapshot);
  by_id("premat_event_count").textContent = total_events > shown_events
    ? `${events.length} shown · ${total_events} total`
    : `${events.length} event${events.length === 1 ? "" : "s"}`;
  by_id("premat_events_body").innerHTML = events.map(event => {
    const transition = `${event.old_state || "—"} → ${event.new_state || event.state || "—"}`;
    const verdict = [event.decision, event.outcome, event.reason].filter(Boolean).join(" · ") || "—";
    const wait = event.wait_ms === null || event.wait_ms === undefined ? "—" : `${Number(event.wait_ms).toFixed(3)} ms`;
    return `<tr><td>${event.sequence ?? "—"}</td><td>${Number(event.elapsed_ms ?? 0).toFixed(3)}</td><td>${event.layer_index ?? "—"}</td><td>${premat_escape(event.family || "—")}</td><td>${premat_escape(transition)}</td><td>${premat_escape(event.headroom_policy || "—")}</td><td>${premat_pair(event.process_allocated_bytes, event.process_ordinary_peak_bytes)}</td><td>${premat_pair(event.device_free_bytes, event.device_used_bytes)}</td><td>${premat_bytes(event.global_buffer_bytes)}</td><td>${premat_bytes(event.premat_headroom_bytes)}</td><td>${premat_bytes(event.predicted_retained_bytes)}</td><td>${premat_bytes(event.predicted_materialisation_peak_bytes)}</td><td>${premat_bytes(event.predicted_foreground_overlap_bytes)}</td><td>${premat_bytes(event.predicted_envelope_bytes)}</td><td>${premat_escape(verdict)}</td><td>${premat_escape(wait)}${event.critical_path_miss ? " · MISS" : ""}</td></tr>`;
  }).join("");
}

function premat_frame_key(payload) {
  const snapshot = payload?.latest;
  if (!snapshot) return null;
  return `${premat_view.run_id || ""}:${snapshot.optimizer_update ?? ""}:${snapshot.latest_event_sequence ?? snapshot.event_count ?? ""}`;
}

// vvv THOG browser-only playback buffers polled display frames.  It never
// changes scheduler capture, the SQLite writer, training cadence, or W&B.
function premat_playback_delay_ms() {
  return Math.max(100, Math.round(PREMAT_PLAYBACK_BASE_MS / premat_view.playback_rate));
}

function premat_schedule_playback() {
  if (premat_view.playback_timer !== null) clearTimeout(premat_view.playback_timer);
  premat_view.playback_timer = setTimeout(() => {
    premat_view.playback_timer = null;
    const payload = premat_view.frame_queue.shift();
    if (payload) render_premat(payload);
    premat_schedule_playback();
  }, premat_playback_delay_ms());
}

function premat_enqueue_frame(payload) {
  const key = premat_frame_key(payload);
  if (key === null) {
    render_premat(payload);
    return;
  }
  if (key === premat_view.newest_frame_key) return;
  premat_view.newest_frame_key = key;
  if (!premat_view.latest_payload) {
    render_premat(payload);
    return;
  }
  premat_view.frame_queue.push(payload);
  if (premat_view.frame_queue.length > PREMAT_PLAYBACK_QUEUE_LIMIT) {
    premat_view.frame_queue.splice(0, premat_view.frame_queue.length - PREMAT_PLAYBACK_QUEUE_LIMIT);
  }
}
// ^^^ THOG

async function refresh_premat() {
  const run_id = app.current_run_id;
  if (!run_id) {
    premat_view.run_id = null;
    premat_view.latest_payload = null;
    premat_view.newest_frame_key = null;
    premat_view.frame_queue = [];
    premat_sync_tab();
    return;
  }
  if (premat_view.run_id !== run_id) {
    premat_view.latest_payload = null;
    premat_view.newest_frame_key = null;
    premat_view.frame_queue = [];
    premat_sync_tab();
  }
  const serial = ++premat_view.request_serial;
  try {
    const payload = await fetch_json(`/api/premat?run=${encodeURIComponent(run_id)}`);
    if (serial !== premat_view.request_serial || run_id !== app.current_run_id) return;
    premat_view.run_id = run_id;
    premat_enqueue_frame(payload);
  } catch (_error) {
    if (serial === premat_view.request_serial) {
      premat_view.latest_payload = null;
      premat_sync_tab();
    }
  }
}

window.premat_refresh = refresh_premat;

window.addEventListener("DOMContentLoaded", () => {
  const playback = by_id("premat_playback_speed");
  const playback_label = by_id("premat_playback_speed_label");
  const update_playback_rate = () => {
    premat_view.playback_rate = 2 ** Number(playback?.value || 0);
    if (playback_label) playback_label.textContent = `${premat_view.playback_rate}×`;
    premat_schedule_playback();
  };
  playback?.addEventListener("input", update_playback_rate);
  update_playback_rate();
  by_id("premat_event_filter")?.addEventListener("input", () => {
    if (premat_view.latest_payload) render_premat(premat_view.latest_payload);
  });
  document.querySelectorAll("[data-premat-sort]").forEach(header => {
    header.addEventListener("click", () => {
      const key = header.dataset.prematSort;
      if (premat_view.sort_key === key) {
        premat_view.sort_descending = !premat_view.sort_descending;
      } else {
        premat_view.sort_key = key;
        premat_view.sort_descending = false;
      }
      if (premat_view.latest_payload) render_premat(premat_view.latest_payload);
    });
  });
  premat_sync_tab();
  refresh_premat();
  setInterval(refresh_premat, 750);
});
// ^^^ THOG
