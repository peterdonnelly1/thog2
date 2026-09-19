"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class FakeElement {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parentElement = null;
    this.hidden = false;
    this.dataset = {};
    this.offsetParent = {};
    this.className = "";
    this.classList = {
      contains: name => this.className.split(/\s+/).includes(name),
    };
  }

  appendChild(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }

  insertBefore(child, reference) {
    if (child.parentElement) child.parentElement.children = child.parentElement.children.filter(item => item !== child);
    const index = this.children.indexOf(reference);
    child.parentElement = this;
    this.children.splice(index < 0 ? this.children.length : index, 0, child);
    return child;
  }

  insertAdjacentElement(position, child) {
    assert.equal(position, "beforebegin");
    return this.parentElement?.insertBefore(child, this) || null;
  }

  remove() {
    if (!this.parentElement) return;
    this.parentElement.children = this.parentElement.children.filter(child => child !== this);
    this.parentElement = null;
  }

  setAttribute(name, value) { this[name] = value; }

  querySelector(selector) {
    if (selector === ".processing-paired-download-groups") {
      return this.children.find(child => child.className === "processing-paired-download-groups") || null;
    }
    if (selector === ".chart-card-actions") return this.actions || null;
    if (selector === ".maximize-button") {
      return this.children.find(child => child.className === "maximize-button") || null;
    }
    return null;
  }

  querySelectorAll(selector) {
    if (selector === ":scope > a") return this.children.filter(child => child.tagName === "A");
    if (selector === ".plot-mount") return this.mounts || [];
    return [];
  }
}

const host = new FakeElement();
host.className = "processing-downloads";
const direct_raw = new FakeElement("a");
direct_raw.id = "processing_download_raw";
direct_raw.textContent = "Raw capture";
host.appendChild(direct_raw);
const direct_compatibility = new FakeElement("a");
direct_compatibility.id = "processing_download_compatibility";
direct_compatibility.textContent = "Compatibility";
host.appendChild(direct_compatibility);
const direct_uncategorized = new FakeElement("span");
direct_uncategorized.textContent = "legacy downloads";
host.appendChild(direct_uncategorized);
const actions = new FakeElement();
const compatibility_card = new FakeElement();
compatibility_card.actions = actions;
const operations_card = new FakeElement();
operations_card.className = "maximized";
const operations_mount = new FakeElement();
operations_mount.dataset.plotReady = "true";
operations_mount.data = [{}, {}];
const elements = new Map([
  ["processing_compatibility_card", compatibility_card],
  ["processing_timeline_card", operations_card],
  ["processing_timeline_plot", operations_mount],
]);
const relayouts = [];
const restyles = [];
const timers = [];
let throughput_renders = 0;
let timing_renders = 0;
let saved_operation_colours = null;

const sandbox = {
  console,
  document:{
    head:{appendChild() {}},
    getElementById() { return null; },
    createElement(tag) { return new FakeElement(tag); },
    addEventListener() {},
    querySelector(selector) {
      if (selector === "#processing_chart_group .processing-downloads") return host;
      return null;
    },
  },
  window:{
    addEventListener() {},
    clearTimeout() {},
    setTimeout(callback) { timers.push(callback); return timers.length; },
    instra_colour_palette:["#111111", "#222222", "#333333"],
  },
  Plotly:{
    relayout:async (mount, update) => { relayouts.push({mount, update}); },
    restyle:async (mount, update, indices) => { restyles.push({mount, update, indices}); },
    Plots:{resize() {}},
  },
  processing_view:{
    resource_available:true,
    timing_entries:[
      {run:{artifact_name:"260916-1405_scruffy_NSYS_PREMAT___FULL_CONFIGURATION"}, run_id:"nsys"},
      {run:{artifact_name:"260916-1440_scruffy_NCU_PREMAT___FULL_CONFIGURATION"}, run_id:"ncu"},
    ],
  },
  processing_resource_panes() {
    return [
      {label:"MAIN", domain:[0.70, 1.0]},
      {label:"PREMAT", domain:[0.35, 0.64]},
      {label:"COMBINED / UNATTRIBUTABLE", domain:[0, 0.29]},
    ];
  },
  processing_escape(value) { return String(value); },
  processing_update_timing_run_name(entry) { return entry.run.artifact_name; },
  processing_gpu_maximize_button(chart_name) {
    const button = new FakeElement("button");
    button.className = "maximize-button";
    button.dataset.maximize = chart_name;
    return button;
  },
  processing_resource_render:async function() {},
  processing_render_contention:async function() {},
  processing_render_update_timing:async function() { timing_renders += 1; },
  processing_render_throughput:async function() { throughput_renders += 1; },
  processing_render:async function() {},
  processing_render_timeline:async function() {},
  processing_gpu_link_time_axes() {},
  processing_throughput_workspace_runs() { return []; },
  render_runs() {},
  run_identifier(run) { return run.dashboard_run_id; },
  is_visible() { return true; },
  by_id(id) { return elements.get(id) || null; },
  encodeURIComponent,
  setTimeout(callback) { timers.push(callback); return timers.length; },
  requestAnimationFrame(callback) { callback(); },
  load_json(_key, fallback) { return fallback; },
  save_json(key, value) {
    saved_operation_colours = {key, value:JSON.parse(JSON.stringify(value))};
  },
};
sandbox.window.processing_view = sandbox.processing_view;

const source = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_processing_user_fixes.js",
  "utf8",
);
const operations_source = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_processing_operations_final.js",
  "utf8",
);
const resource_source = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_processing_resource_attribution.js",
  "utf8",
);
vm.runInNewContext(source, sandbox);
const hooks = sandbox.window.processing_user_fix_test_hooks;
vm.runInNewContext(operations_source, sandbox);
const operation_hooks = sandbox.window.processing_operations_test_hooks;

const guide_model = operation_hooks.layer_guides([
  {owner:"MAIN", layer:0, start_us:10000},
  {owner:"MAIN", layer:1, start_us:20000},
  {owner:"MAIN", layer:2, start_us:35000},
]);
assert.equal(guide_model.shapes.length, 3, "each layer should have one continuous divider");
assert.deepEqual(Array.from(guide_model.shapes, shape => [shape.y0, shape.y1]), [
  [0.28, 1.42], [0.28, 1.42], [0.28, 1.42],
]);
assert.equal(guide_model.annotations.length, 6, "layer numbers should be duplicated above and below the lanes");
assert.deepEqual(Array.from(guide_model.annotations.slice(0, 2), annotation => [annotation.name, annotation.y]), [
  ["processing-layer-0-top", 1.42], ["processing-layer-0-bottom", 0.28],
]);
assert.deepEqual(
  Array.from(operation_hooks.layer_zoom_range(guide_model.sorted, 50, 1), value => Number(value.toFixed(1))),
  [18.8, 36.8],
  "layer zoom did not retain 12% neighbour context",
);
const scroll_metrics = operation_hooks.layer_scroll_metrics(100, [20, 40], 500);
assert.equal(scroll_metrics.virtual_width, 2500, "zoom scrollbar does not represent the full capture");
assert.equal(scroll_metrics.max_scroll, 2000);
assert.equal(scroll_metrics.scroll_left, 500);
assert.deepEqual(
  Array.from(operation_hooks.layer_range_for_scroll(100, 20, 1000, 2000), value => Number(value.toFixed(1))),
  [40, 60],
  "scroll position did not pan the fixed-width zoom window",
);
assert.match(operations_source, /processing_operations_reset_zoom/);
assert.match(operations_source, /training_throughput_plot/);
assert.match(operations_source, /processing_render_throughput_before_training_group/);
assert.match(operations_source, /processing_view\.training_throughput_available/);
assert.doesNotMatch(
  operations_source,
  /training_chart_group[\s\S]{0,1200}processing_view\.available/,
  "Training throughput visibility still depends on Processing availability",
);
assert.match(operations_source, /#training_chart_group\s*\{[\s\S]*?min-height:0;/);
assert.match(operations_source, /processing\.insertAdjacentElement\("afterend", group\)/);
const chart_groups = new FakeElement("main");
const training_group = new FakeElement("section");
const processing_group = new FakeElement("section");
elements.set("training_chart_group", training_group);
elements.set("processing_chart_group", processing_group);
chart_groups.appendChild(training_group);
chart_groups.appendChild(processing_group);
sandbox.processing_view.charts_tab_visible = true;
sandbox.processing_view.available = true;
sandbox.processing_view.training_throughput_available = true;
sandbox.processing_view.trace_available = true;
operation_hooks.sync_training_group_presentation();
assert.deepEqual(chart_groups.children, [processing_group, training_group],
  "a selected profiler trace must be presented before the live Training mirror");
assert.equal(training_group.hidden, false);
sandbox.processing_view.trace_available = false;
operation_hooks.sync_training_group_presentation();
assert.deepEqual(chart_groups.children, [training_group, processing_group],
  "a throughput-only run should retain the Training-first presentation");

const headings = hooks.resource_heading_annotations({stream_resources:[{}]});
assert.deepEqual(Array.from(headings, item => item.text), [
  "<b>MAIN</b>", "<b>PREMAT</b>", "<b>COMBINED / UNATTRIBUTABLE</b>",
]);

const names = hooks.timing_name_annotations();
assert.equal(names.length, 2);
assert.match(names[0].text, /FULL_CONFIGURATION/);
assert.equal(names[0].y, 0.36);
assert.ok(Math.abs(names[1].y - 1.36) < 1e-9);

const files = hooks.download_entries({json:"compat.json", bundle:"bundle.zip", duplicate:"bundle.zip"});
assert.deepEqual(Array.from(files, item => item[1]), ["bundle.zip", "compat.json"]);
assert.match(hooks.processing_download_url_for_run("ncu id", "compat.json"), /run=ncu%20id/);
const evidence_files = hooks.download_entries({
  raw_trace:"trace.nsys-rep", metric_audit:"audit.csv", everything:"everything.zip",
});
assert.deepEqual(Array.from(evidence_files, item => item[0]), ["everything", "metric_audit", "raw_trace"]);
hooks.decorate_direct_downloads();
assert.match(direct_raw.title, /Original Nsight Systems/);
assert.match(direct_compatibility.title, /structural compatibility catalogue/);

assert.match(resource_source, /Active-SM allocated warp slots/);
assert.match(resource_source, /Idle-SM warp-slot headroom/);
assert.match(resource_source, /Nsight metric:/);
assert.match(resource_source, /processing_gpu_link_inspection\(payload\)/);
assert.match(resource_source, /payload\.premat_lifecycle/);

hooks.render_paired_downloads({
  paired_processing_downloads:{
    pair:{dashboard_run_id:"nsys", artifact_name:"pair", files:{everything:"everything.zip", pair_manifest:"manifest.json"}},
    nsys:{dashboard_run_id:"nsys", artifact_name:"NSYS full", files:{bundle:"nsys.zip", raw_trace:"trace.nsys-rep"}},
    ncu:{dashboard_run_id:"ncu", artifact_name:"NCU full", files:{
      raw_ncu:"trace.ncu-rep", ncu_raw_csv:"raw.csv", ncu_semantic_csv:"semantic.csv",
      kernel_resources:"resources.csv", json:"compat.json",
    }},
  },
});
const rendered_groups = host.querySelector(".processing-paired-download-groups");
assert.ok(rendered_groups);
assert.equal(rendered_groups.children.length, 3);
assert.equal(rendered_groups.children[0].children[0].tagName, "STRONG");
assert.equal(rendered_groups.children[0].children[0].textContent, "PAIR:");
assert.equal(rendered_groups.children[1].children[0].textContent, "NSYS:");
assert.equal(rendered_groups.children[2].children[0].textContent, "NCU:");
assert.equal(direct_raw.hidden, true);
assert.equal(direct_compatibility.hidden, true);
assert.equal(direct_uncategorized.hidden, true, "an uncategorized legacy download item remained visible beside paired groups");
assert.equal(rendered_groups.children[0].children[1].textContent, "Everything");
assert.match(rendered_groups.children[1].children[1].href, /run=nsys/);
assert.match(rendered_groups.children[2].children[1].href, /run=ncu/);
assert.equal(rendered_groups.children[1].children.at(-1).textContent, "Raw");
assert.equal(rendered_groups.children[2].children[1].textContent, "Raw");
assert.equal(rendered_groups.children[2].children[2].textContent, "Metrics");
assert.equal(rendered_groups.children[2].children[3].textContent, "Semantic");
assert.match(rendered_groups.children[2].children[1].title, /Original Nsight Compute/);
assert.match(rendered_groups.children[1].children.at(-1).title, /Original Nsight Systems/);
hooks.render_paired_downloads({});
assert.equal(direct_raw.hidden, false, "unpairing did not restore the run-owned download buttons");
assert.equal(direct_uncategorized.hidden, false, "unpairing did not restore the original download-bar contents");

assert.match(operations_source, /const lane_width = 0\.30/);
assert.match(operations_source, /#processing_timeline_card:not\(\.maximized\)[\s\S]*?height:300px !important/,
  "the normal Operations card does not leave room for the outer layer guides");
assert.match(operations_source, /captureevents:true/);
assert.match(operations_source, /plotly_clickannotation/);
assert.match(operations_source, /previous_span.*0\.12/s);
assert.equal(operation_hooks.marker_for({owner:"MAIN", operation:"consume", family:"QKV"}).pattern, undefined,
  "MAIN GEMM operations should use a plain fill");
assert.equal(operation_hooks.marker_for({owner:"MAIN", operation:"materialize", family:"QKV"}).pattern.shape, "/",
  "MAIN materialisation operations should carry the diagonal pattern");
assert.equal(operation_hooks.marker_for({owner:"PREMAT", operation:"materialize", family:"QKV"}).pattern.shape, "/",
  "PREMAT operations should carry the diagonal pattern");
assert.equal(
  operation_hooks.operation_colour_key({owner:"MAIN", operation:"consume", family:"QKV"}),
  operation_hooks.operation_colour_key({owner:"PREMAT", operation:"materialize", family:"QKV"}),
  "MAIN GEMM and PREMAT did not share their matrix-family colour identity",
);
assert.deepEqual(Array.from(operation_hooks.operations_colour_palette()), ["#111111", "#222222", "#333333"],
  "Operations did not reuse the Runs colour-patch palette");
assert.match(operations_source, /toggle\.textContent = "PREMAT"/);
assert.match(operations_source, /thog2_processing_operation_colours_v1/);

hooks.apply_operations_maximized_geometry();
assert.equal(restyles.at(-1).update.width, 0.30);
assert.deepEqual(Array.from(relayouts.at(-1).update["yaxis.range"]), [0.20, 1.50]);
operations_card.className = "";
hooks.apply_operations_maximized_geometry();
assert.equal(restyles.at(-1).update.width, 0.30);
assert.deepEqual(Array.from(relayouts.at(-1).update["yaxis.range"]), [0.20, 1.50]);

sandbox.processing_render({}, true).then(async () => {
  assert.equal(actions.children[0].dataset.maximize, "processing_compatibility");
  timers.length = 0;
  sandbox.processing_view.throughput_last_payload = {throughput:[]};
  sandbox.processing_throughput_workspace_runs = () => [
    {dashboard_run_id:"new-run", revision:"r1", run_state:"finished"},
  ];
  sandbox.render_runs();
  assert.equal(timers.length, 1, "run-list changes did not queue comparison refresh");
  await timers[0]();
  assert.equal(throughput_renders, 1, "new visible run did not refresh throughput");
  assert.equal(timing_renders, 1, "new visible run did not refresh aligned timing");
  timers.length = 0;
  sandbox.render_runs();
  await timers[0]();
  assert.equal(throughput_renders, 1, "unchanged run cohort repeated throughput work");
  assert.equal(timing_renders, 1, "unchanged run cohort repeated timing work");
  const colour_row = {owner:"MAIN", operation:"consume", family:"QKV"};
  const colour_key = operation_hooks.operation_colour_key(colour_row);
  operations_mount.data = [
    {meta:{operations_colour_key:colour_key, operations_owner:"MAIN"}},
    {meta:{operations_colour_key:colour_key, operations_owner:"PREMAT"}},
  ];
  await operation_hooks.apply_operation_colour(colour_key, "#abcdef");
  assert.equal(saved_operation_colours.key, "thog2_processing_operation_colours_v1");
  assert.equal(saved_operation_colours.value[colour_key], "#ABCDEF",
    "the chosen operation colour was not persisted");
  assert.deepEqual(Array.from(restyles.at(-1).indices), [0, 1]);
  assert.equal(restyles.at(-1).update["marker.color"], "#ABCDEF",
    "the chosen operation colour was not applied to the live trace");
  assert.equal(restyles.at(-1).update["marker.pattern.bgcolor"], "#ABCDEF",
    "the chosen operation colour was not applied beneath the PREMAT stripe pattern");
  operations_mount._processing_layer_zoom_capture_ms = 100;
  operations_mount._processing_layer_zoom_range = [20, 40];
  await operation_hooks.reset_layer_zoom(operations_mount);
  assert.deepEqual(Array.from(relayouts.at(-1).update["xaxis.range"]), [0, 100],
    "Reset Zoom did not restore the full capture range");
  assert.equal(relayouts.at(-1).update["xaxis.autorange"], false);
  console.log("PASS paired downloads, resource headings, maximize controls, dynamic comparison refresh and compact chart geometry");
}).catch(error => {
  console.error(error);
  process.exitCode = 1;
});
