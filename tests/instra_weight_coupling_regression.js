// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

class FakeElement {
  constructor({id = "", chart_name = null, text = ""} = {}) {
    this.id = id;
    this.dataset = chart_name ? {chart: chart_name} : {};
    this.textContent = text;
    this.hidden = false;
    this.title = "";
    this.parentElement = null;
    this.children = [];
    this.attributes = new Map();
    this.strong_child = null;
  }

  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.has(name) ? this.attributes.get(name) : null; }
  hasAttribute(name) { return this.attributes.has(name); }
  removeAttribute(name) { this.attributes.delete(name); }

  appendChild(child) {
    if (child.parentElement) {
      const siblings = child.parentElement.children;
      const index = siblings.indexOf(child);
      if (index >= 0) siblings.splice(index, 1);
    }
    child.parentElement = this;
    this.children.push(child);
    if (child.tagName === "STRONG") {
      this.strong_child = child;
      this.textContent += child.textContent || "";
    }
    return child;
  }

  replaceChildren(...nodes) {
    this.children = nodes;
    this.strong_child = nodes.find(node => node.tagName === "STRONG") || null;
    this.textContent = nodes.map(node => node.textContent || "").join("");
  }

  querySelector(selector) {
    if (selector === "strong") return this.strong_child;
    const match = selector.match(/data-chart="([^"]+)"/);
    if (match) return this.children.find(child => child.dataset?.chart === match[1]) || null;
    return null;
  }

  querySelectorAll(selector) {
    if (selector.includes(".chart-card")) return this.children.filter(child => child.dataset?.chart);
    return [];
  }

  matches(selector) { return selector === ".chart-card" && Boolean(this.dataset?.chart); }
}

const repository_root = path.resolve(__dirname, "..");
const presentation_path = path.join(
  repository_root,
  "sheet/local_dashboard_assets/dashboard_weight_coupling_presentation_patch.js",
);
const reliability_path = path.join(
  repository_root,
  "sheet/local_dashboard_assets/dashboard_weight_coupling_reliability_patch.js",
);

const ids = new Map();
const grid = new FakeElement({id: "chart_grid"});
ids.set("chart_grid", grid);
ids.set("weight_index_group_controls", new FakeElement({id: "weight_index_group_controls"}));
ids.set("weight_index_group_summary", new FakeElement({id: "weight_index_group_summary"}));
for (const id of [
  "weight_residual_minus",
  "weight_residual_plus",
  "weight_branch_minus",
  "weight_branch_plus",
]) {
  ids.set(id, new FakeElement({id}));
}
const random_button = new FakeElement({id: "weight_random_jump", text: "random"});
random_button.hidden = true;
random_button.setAttribute("hidden", "");
ids.set("weight_random_jump", random_button);

const chart_names = [
  "attn_q_head_N",
  "attn_k_head_N",
  "attn_v_head_N",
  "attn_out_head_N",
  "mlp_up",
  "mlp_down",
];
const initial_order = [
  "attn_q_head_N",
  "attn_k_head_N",
  "attn_v_head_N",
  "attn_out_head_N",
  "mlp_up",
  "mlp_down",
];
const headings = new Map();
const initial_titles = {
  attn_q_head_N: "Attention query scalar trajectories",
  attn_k_head_N: "Attention key scalar trajectories",
  attn_v_head_N: "Attention value scalar trajectories",
  attn_out_head_N: "Attention output scalar trajectories",
  mlp_up: "MLP expansion scalar trajectories",
  mlp_down: "MLP contraction scalar trajectories",
};
for (const chart_name of initial_order) {
  const card = new FakeElement({chart_name});
  grid.appendChild(card);
  headings.set(chart_name, new FakeElement({text: initial_titles[chart_name]}));
  ids.set(`${chart_name}_plot`, new FakeElement({id: `${chart_name}_plot`}));
}

const intervals = new Map();
let next_interval = 1;
let mutation_observer_count = 0;

const document = {
  head: new FakeElement({id: "head"}),
  createElement(tag) {
    const element = new FakeElement();
    element.tagName = String(tag).toUpperCase();
    return element;
  },
  createTextNode(text) { return {textContent: String(text), tagName: "#TEXT"}; },
  querySelector(selector) {
    const chart_name = selector.match(/\.chart-card\[data-chart="([^"]+)"\]/)?.[1];
    if (!chart_name) return null;
    if (selector.includes(".chart-heading-copy h2")) return headings.get(chart_name) || null;
    return grid.children.find(child => child.dataset?.chart === chart_name) || null;
  },
};

const deep_clone = value => JSON.parse(JSON.stringify(value));
const trace_kind = trace => trace?.meta?.instra_weight_selection_kind || null;
const matches_selected = trace => (
  ["user", "user_random"].includes(trace_kind(trace))
  && trace.meta.instra_weight_model_feature === 123
  && trace.meta.instra_weight_intermediate_feature === 145
);

const figure_map = Object.fromEntries(
  chart_names.map(chart_name => [chart_name, {data: [], layout: {title: {text: ""}}}]),
);
const chart_titles = {...initial_titles};
const context = {
  console,
  structuredClone: global.structuredClone,
  window: {
    addEventListener(event, callback) { if (event === "load") callback(); },
    __instra_matched_weight_selection: {
      selection: () => ({user_selected: true, model_feature: 123, intermediate_feature: 145}),
      capability: () => ({available: true, maximum: 767, reason: ""}),
    },
  },
  document,
  chart_titles,
  app: {workspace_mode: false, chart_settings_render_override: null},
  by_id: id => ids.get(id) || null,
  figure_for_chart: chart_name => figure_map[chart_name] || null,
  normalize_chart_settings: (chart_name, supplied = null) => ({
    title: supplied?.title || chart_titles[chart_name] || chart_name,
    current_weights_only: true,
    join_with_line_segments: true,
    ...(supplied || {}),
  }),
  prepare_figure: figure => {
    const prepared = deep_clone(figure);
    for (const trace of prepared.data || []) {
      if (trace?.meta?.instra_thog_weight !== true) continue;
      trace.x = [...trace.meta.instra_thog_integer_x];
      trace.y = [...trace.meta.instra_thog_integer_y];
      trace.mode = "lines+markers";
      trace.line = {...(trace.line || {}), shape: "linear"};
    }
    if (figure.__simulate_matched_filter) prepared.data = prepared.data.filter(matches_selected);
    return prepared;
  },
  ensure_depth_cards: () => undefined,
  show_toast: message => { context.last_toast = String(message); },
  setTimeout: callback => { callback(); return 1; },
  clearTimeout: () => undefined,
  setInterval: callback => {
    const id = next_interval++;
    intervals.set(id, {callback, active: true});
    return id;
  },
  clearInterval: id => {
    const interval = intervals.get(id);
    if (interval) interval.active = false;
  },
  requestAnimationFrame: () => 1,
  MutationObserver: class {
    constructor() {
      mutation_observer_count += 1;
      throw new Error("weight presentation patches must not install a MutationObserver");
    }
  },
};
context.window.window = context.window;
vm.createContext(context);

for (const source_path of [presentation_path, reliability_path]) {
  vm.runInContext(fs.readFileSync(source_path, "utf8"), context, {filename: source_path});
}

for (const [id, interval] of intervals) {
  if (interval.active) interval.callback();
  assert.equal(interval.active, false, `startup interval ${id} did not terminate when the DOM was ready`);
}
assert.equal(mutation_observer_count, 0, "a persistent MutationObserver was installed");
assert.deepEqual(
  grid.children.map(card => card.dataset.chart),
  [
    "attn_q_head_N",
    "attn_k_head_N",
    "mlp_up",
    "attn_v_head_N",
    "attn_out_head_N",
    "mlp_down",
  ],
  "weight card order is wrong",
);
assert.equal(random_button.hidden, false, "random control remained hidden");
assert.equal(random_button.hasAttribute("hidden"), false, "random hidden attribute remained");
assert.equal(random_button.textContent, "random");

for (const [chart_name, letter] of Object.entries({
  attn_q_head_N: "Q",
  attn_k_head_N: "K",
  attn_v_head_N: "V",
  attn_out_head_N: "O",
})) {
  assert.equal(headings.get(chart_name).textContent, `Attention - ${letter}`);
  assert.equal(headings.get(chart_name).strong_child?.textContent, letter);
}

const random_trace = (model_feature, intermediate_feature, width = 3.6) => ({
  mode: "lines",
  x: [1, 1.5, 2],
  y: [0.1, 0.3, 0.2],
  showlegend: true,
  name: `residual feature ${model_feature} · branch feature ${intermediate_feature}`,
  hovertemplate: `residual feature ${model_feature} · branch feature ${intermediate_feature}`,
  line: {width, shape: "spline"},
  meta: {
    instra_thog_weight: true,
    instra_thog_integer_x: [1, 2],
    instra_thog_integer_y: [0.1, 0.2],
    instra_weight_selection_protocol: "matched_six_v1",
    instra_weight_selection_kind: "random",
    instra_weight_model_feature: model_feature,
    instra_weight_intermediate_feature: intermediate_feature,
  },
});
const selected_trace = () => ({
  mode: "lines",
  x: [1, 1.5, 2],
  y: [0.4, 0.7, 0.5],
  showlegend: true,
  name: "residual feature 123 · branch feature 145",
  hovertemplate: "residual feature 123 · branch feature 145",
  line: {width: 3.6, shape: "spline"},
  meta: {
    instra_thog_weight: true,
    instra_thog_integer_x: [1, 2],
    instra_thog_integer_y: [0.4, 0.5],
    instra_weight_selection_protocol: "matched_six_v1",
    instra_weight_selection_kind: "user",
    instra_weight_model_feature: 123,
    instra_weight_intermediate_feature: 145,
  },
});

let figure = {
  __simulate_matched_filter: true,
  data: [random_trace(2, 3)],
  layout: {
    title: {text: "DEPTH generated scalar trajectories — attention query<br><sup>subtitle</sup>"},
  },
};
let prepared = context.prepare_figure(figure, "attn_q_head_N");
assert.equal(prepared.data.length, 0, "a differently indexed random coupling leaked into the selected view");
assert.equal(prepared.layout.title, undefined, "the redundant Plotly title was retained");
assert.equal(prepared.layout.showlegend, false, "the redundant right-hand legend was retained");
assert.equal(prepared.layout.legend, undefined);
assert.match(
  prepared.layout.annotations?.[0]?.text || "",
  /selected coupling 123 → 145 was not recorded/i,
);

figure = {
  __simulate_matched_filter: true,
  data: [random_trace(2, 3)],
  layout: {title: {text: "DEPTH generated scalar trajectories — MLP contraction<br><sup>subtitle</sup>"}},
};
prepared = context.prepare_figure(figure, "mlp_down");
assert.equal(prepared.data.length, 0);
assert.match(prepared.layout.annotations?.[0]?.text || "", /123 → 145/);

figure = {
  __simulate_matched_filter: true,
  data: [random_trace(2, 3), selected_trace()],
  layout: {title: {text: "DEPTH generated scalar trajectories — attention value<br><sup>subtitle</sup>"}},
};
prepared = context.prepare_figure(figure, "attn_v_head_N");
assert.equal(prepared.data.length, 1, "selected coupling should remain singular");
assert.notEqual(prepared.data[0].meta.instra_weight_selection_fallback, true);
assert.ok(Math.abs(prepared.data[0].line.width - 2.88) < 1e-12);
assert.match(prepared.data[0].name, /input feature 123 → output feature 145/);
assert.deepEqual(prepared.data[0].x, [1, 2]);
assert.deepEqual(prepared.data[0].y, [0.4, 0.5]);
assert.equal(prepared.data[0].line.shape, "linear");
assert.equal(prepared.data[0].showlegend, false);
assert.equal(prepared.layout.title, undefined);
assert.equal(prepared.layout.showlegend, false);
assert.ok(!(prepared.layout.annotations || []).some(annotation => /not recorded/i.test(annotation.text || "")));

context.app.workspace_mode = true;
figure = {
  __simulate_matched_filter: true,
  data: [random_trace(9, 10)],
  layout: {title: {text: "DEPTH generated scalar trajectories — attention key"}},
};
prepared = context.prepare_figure(figure, "attn_k_head_N");
assert.equal(prepared.data.length, 0, "Runs-only fallback leaked into Workspace");
assert.match(prepared.layout.annotations?.[0]?.text || "", /123 → 145/);

console.log("instra weight coupling regression: PASS");
// ^^^ THOG
