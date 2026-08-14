// vvv THOG
"use strict";

// W&B-like per-artifact navigation and Overview rendering for the local dashboard.
// Charts remains the default view; Logs, Files and Artifacts are intentionally blank for now.

const local_detail_tabs = Object.freeze(["charts", "overview", "logs", "files", "artifacts"]);
let local_active_detail_tab = "charts";

function local_format_overview_timestamp(value) {
  if (!value) return "—";
  const timestamp = new Date(value);
  if (!Number.isFinite(timestamp.getTime())) return String(value);
  return timestamp.toLocaleString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

function local_format_duration(start_value, end_value, running) {
  const start = Date.parse(start_value || "");
  const end = running ? Date.now() : Date.parse(end_value || "");
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return "—";
  let seconds = Math.max(0, Math.floor((end - start) / 1000));
  const days = Math.floor(seconds / 86400);
  seconds -= days * 86400;
  const hours = Math.floor(seconds / 3600);
  seconds -= hours * 3600;
  const minutes = Math.floor(seconds / 60);
  seconds -= minutes * 60;
  const parts = [];
  if (days) parts.push(`${days}d`);
  if (days || hours) parts.push(`${hours}h`);
  if (days || hours || minutes) parts.push(`${minutes}m`);
  parts.push(`${seconds}s`);
  return parts.join(" ");
}

function local_wandb_owner(run) {
  try {
    const parts = new URL(run?.wandb_url || "").pathname.split("/").filter(Boolean);
    return parts[0] || "—";
  } catch (_error) {
    return "—";
  }
}

function local_wandb_run_path(run) {
  try {
    const parts = new URL(run?.wandb_url || "").pathname.split("/").filter(Boolean);
    if (parts.length >= 4 && parts[2] === "runs") return `${parts[0]}/${parts[1]}/${parts[3]}`;
  } catch (_error) {
    // Fall back to the local run directory below.
  }
  return run?.run_directory || "—";
}

function local_first_present(object, keys, fallback = "—") {
  for (const key of keys) {
    const value = object?.[key];
    if (value !== null && value !== undefined && value !== "") return value;
  }
  return fallback;
}

function local_overview_display_value(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : String(value);
  if (Array.isArray(value)) return value.length ? value.join(", ") : "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function local_state_badge(run) {
  const state = display_run_state(run);
  const badge = document.createElement("span");
  badge.className = `state-badge ${state}`;
  const icon = document.createElement("span");
  icon.className = "state-icon";
  icon.appendChild(icon_svg(["running", "finished", "crashed"].includes(state) ? state : "unknown"));
  badge.append(icon, document.createTextNode(state));
  return badge;
}

function local_append_overview_row(container, label_text, value) {
  const label = document.createElement("div");
  label.className = "overview-meta-label";
  label.textContent = label_text;
  const body = document.createElement("div");
  body.className = "overview-meta-value";
  if (value instanceof Node) body.appendChild(value);
  else body.textContent = local_overview_display_value(value);
  container.append(label, body);
}

function local_hardware_block(configuration) {
  const block = document.createElement("div");
  block.className = "overview-hardware-grid";
  const rows = [
    ["CPU count", local_first_present(configuration, ["cpu_count", "physical_cpu_count"])],
    ["Logical CPU count", local_first_present(configuration, ["logical_cpu_count", "cpu_logical_count"])],
    ["GPU count", local_first_present(configuration, ["gpu_count"])],
    ["GPU type", local_first_present(configuration, ["gpu_type", "gpu_name"])],
  ];
  for (const [label_text, value] of rows) {
    const label = document.createElement("span");
    label.textContent = label_text;
    const body = document.createElement("span");
    body.textContent = local_overview_display_value(value);
    block.append(label, body);
  }
  return block;
}

function local_object_summary(value) {
  if (Array.isArray(value)) return `[ ${value.length} items ]`;
  if (value && typeof value === "object") return `{ ${Object.keys(value).length} keys }`;
  return null;
}

function local_config_value_node(value) {
  const object_summary = local_object_summary(value);
  if (!object_summary) {
    const span = document.createElement("span");
    span.textContent = local_overview_display_value(value);
    return span;
  }
  const details = document.createElement("details");
  details.className = "overview-object-details";
  const summary = document.createElement("summary");
  summary.textContent = object_summary;
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(value, null, 2);
  details.append(summary, pre);
  return details;
}

function local_compile_filter(query) {
  const trimmed = query.trim();
  if (!trimmed) return () => true;
  try {
    const regex = new RegExp(trimmed, "i");
    return text => regex.test(text);
  } catch (_error) {
    const lowered = trimmed.toLowerCase();
    return text => text.toLowerCase().includes(lowered);
  }
}

function local_render_key_panel(container, title_text, noun, values) {
  container.replaceChildren();
  const heading = document.createElement("div");
  heading.className = "overview-panel-heading";
  const heading_text = document.createElement("h3");
  heading_text.textContent = title_text;
  const count = document.createElement("span");
  count.textContent = `${Object.keys(values).length} keys`;
  heading.append(heading_text, count);

  const search = document.createElement("label");
  search.className = "overview-search";
  const icon = document.createElement("span");
  icon.textContent = "⌕";
  icon.setAttribute("aria-hidden", "true");
  const input = document.createElement("input");
  input.type = "search";
  input.placeholder = "Search keys with regex";
  input.setAttribute("aria-label", `Search ${noun} keys with regex`);
  search.append(icon, input);

  const rows = document.createElement("div");
  rows.className = "overview-key-rows";
  const entries = Object.entries(values).sort(([left], [right]) => left.localeCompare(right));
  for (const [key, value] of entries) {
    const row = document.createElement("div");
    row.className = "overview-key-row";
    const key_node = document.createElement("div");
    key_node.className = "overview-key-name";
    key_node.textContent = key;
    const value_node = document.createElement("div");
    value_node.className = "overview-key-value";
    value_node.appendChild(local_config_value_node(value));
    row.dataset.searchText = `${key} ${local_overview_display_value(value)}`;
    row.append(key_node, value_node);
    rows.appendChild(row);
  }
  input.addEventListener("input", () => {
    const accepts = local_compile_filter(input.value);
    for (const row of rows.children) row.hidden = !accepts(row.dataset.searchText || "");
  });
  container.append(heading, search, rows);
}

function local_summary_metrics(run, configuration) {
  return {
    artifact_name: run?.artifact_name ?? "—",
    artifact_prefix: local_first_present(configuration, ["artifact_prefix"]),
    comparison_group: local_first_present(configuration, ["comparison_group", "experiment_prefix", "group"]),
    dense_equivalent_parameters: local_first_present(configuration, ["dense_equivalent_parameters", "dense_equivalent_total_parameters"]),
    model_type: local_first_present(configuration, ["model_type"], run?.model_type || "—"),
    persistent_parameters: local_first_present(configuration, ["persistent_parameters"]),
    "train/loss": local_first_present(configuration, ["train/loss", "train_loss", "training_loss"]),
    "val/val_loss": local_first_present(configuration, ["val/val_loss", "validation_loss", "val_loss"]),
  };
}

function local_render_artifact_outputs(container, run) {
  container.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = "Artifact Outputs";
  const note = document.createElement("p");
  note.className = "overview-artifact-note";
  note.textContent = "This run produced these artifacts as outputs. Total: 1.";
  const table = document.createElement("table");
  table.className = "overview-artifact-table";
  const head = document.createElement("thead");
  head.innerHTML = "<tr><th>Type</th><th>Name</th><th>Size</th><th>Consumer count</th></tr>";
  const body = document.createElement("tbody");
  const row = document.createElement("tr");
  const values = ["local-charts", "charts.sqlite3", format_bytes(run?.database_bytes), "—"];
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.appendChild(cell);
  }
  row.title = run?.run_directory || "";
  body.appendChild(row);
  table.append(head, body);
  container.append(heading, note, table);
}

function local_render_overview() {
  const pane = by_id("run_overview_pane");
  const run = current_run();
  if (!pane || !run) return;
  const configuration = run.configuration || {};

  const metadata = by_id("overview_metadata");
  metadata.replaceChildren();
  local_append_overview_row(metadata, "Notes", local_first_present(configuration, ["notes", "note"]));
  local_append_overview_row(metadata, "Tags", local_first_present(configuration, ["tags"]));
  local_append_overview_row(metadata, "Author", local_first_present(configuration, ["author", "wandb_entity"], local_wandb_owner(run)));
  local_append_overview_row(metadata, "State", local_state_badge(run));
  local_append_overview_row(metadata, "Start time", local_format_overview_timestamp(run.created_at));
  local_append_overview_row(
    metadata,
    "Runtime",
    local_format_duration(run.created_at, run.updated_at, display_run_state(run) === "running"),
  );
  local_append_overview_row(metadata, "Run path", local_wandb_run_path(run));
  local_append_overview_row(metadata, "Hostname", local_first_present(configuration, ["hostname", "host"], run.host_label || "—"));
  local_append_overview_row(metadata, "OS", local_first_present(configuration, ["os", "platform", "platform_string"]));
  local_append_overview_row(metadata, "Python version", local_first_present(configuration, ["python_version"]));
  local_append_overview_row(metadata, "Git repository", local_first_present(configuration, ["git_repository", "repository", "repo"]));
  local_append_overview_row(metadata, "Git state", local_first_present(configuration, ["git_state", "git_commit", "git_hash", "commit_hash"]));
  local_append_overview_row(metadata, "Python executable", local_first_present(configuration, ["python_executable"]));
  local_append_overview_row(metadata, "Command", local_first_present(configuration, ["command", "run_command"]));
  local_append_overview_row(metadata, "System Hardware", local_hardware_block(configuration));
  local_append_overview_row(metadata, "W&B CLI Version", local_first_present(configuration, ["wandb_version", "wandb_cli_version"]));
  local_append_overview_row(metadata, "Group", local_first_present(configuration, ["comparison_group", "experiment_prefix", "group"]));
  local_append_overview_row(metadata, "Job Type", local_first_present(configuration, ["job_type"], run.model_type || "—"));

  local_render_key_panel(by_id("overview_config_panel"), "Config", "config", configuration);
  local_render_key_panel(by_id("overview_summary_panel"), "Summary", "summary", local_summary_metrics(run, configuration));
  local_render_artifact_outputs(by_id("overview_artifact_outputs"), run);
}

function local_apply_detail_tab() {
  const has_run = Boolean(app.current_run_id);
  const tabs = by_id("run_detail_tabs");
  if (tabs) tabs.hidden = !has_run;
  document.querySelectorAll(".run-detail-tab").forEach(button => {
    const active = button.dataset.detailTab === local_active_detail_tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });

  const charts_selected = local_active_detail_tab === "charts";
  const overview_selected = local_active_detail_tab === "overview";
  const blank_selected = !charts_selected && !overview_selected;
  by_id("charts_empty").hidden = has_run || !charts_selected;
  by_id("charts_scroll").hidden = !has_run || !charts_selected;
  by_id("run_overview_pane").hidden = !has_run || !overview_selected;
  by_id("run_blank_detail_pane").hidden = !has_run || !blank_selected;

  if (overview_selected && has_run) local_render_overview();
  if (charts_selected && has_run) {
    requestAnimationFrame(() => requestAnimationFrame(resize_visible_plots));
  }
}

function local_set_detail_tab(tab_name) {
  if (!local_detail_tabs.includes(tab_name)) return;
  if (app.maximized_chart) restore_maximized_chart();
  local_active_detail_tab = tab_name;
  local_apply_detail_tab();
}

function local_install_detail_tabs() {
  const toolbar = document.querySelector(".charts-toolbar");
  if (!toolbar || by_id("run_detail_tabs")) return;

  const tabs = document.createElement("nav");
  tabs.id = "run_detail_tabs";
  tabs.className = "run-detail-tabs";
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "Artifact views");
  for (const tab_name of local_detail_tabs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "run-detail-tab";
    button.dataset.detailTab = tab_name;
    button.setAttribute("role", "tab");
    button.textContent = tab_name[0].toUpperCase() + tab_name.slice(1);
    button.addEventListener("click", () => local_set_detail_tab(tab_name));
    tabs.appendChild(button);
  }
  toolbar.insertAdjacentElement("afterend", tabs);

  const overview = document.createElement("section");
  overview.id = "run_overview_pane";
  overview.className = "run-overview-pane";
  overview.hidden = true;
  overview.innerHTML = `
    <div class="overview-metadata" id="overview_metadata"></div>
    <div class="overview-data-grid">
      <section class="overview-key-panel" id="overview_config_panel"></section>
      <section class="overview-key-panel" id="overview_summary_panel"></section>
    </div>
    <section class="overview-artifact-outputs" id="overview_artifact_outputs"></section>
  `;

  const blank = document.createElement("section");
  blank.id = "run_blank_detail_pane";
  blank.className = "run-blank-detail-pane";
  blank.hidden = true;
  by_id("charts_pane").append(overview, blank);

  const stylesheet = document.createElement("link");
  stylesheet.rel = "stylesheet";
  stylesheet.href = "/assets/dashboard_overview_patch.css";
  document.head.appendChild(stylesheet);
  local_apply_detail_tab();
}

const local_base_render_run_heading = render_run_heading;
render_run_heading = function() {
  local_base_render_run_heading();
  local_apply_detail_tab();
};

const local_base_render_empty_state = render_empty_state;
render_empty_state = function() {
  local_base_render_empty_state();
  local_apply_detail_tab();
};

const local_base_select_run = select_run;
select_run = function(run_id, options = {}) {
  local_active_detail_tab = "charts";
  return local_base_select_run(run_id, options);
};

local_install_detail_tabs();
// ^^^ THOG
