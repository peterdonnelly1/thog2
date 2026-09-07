// vvv THOG
"use strict";

// Lightweight browser-regression seam: preserve Plotly's DOM/data/event contract
// without requiring WebGL or a separately installed plotly.js bundle.
(function(global) {
  const install_events = mount => {
    if (typeof mount.on === "function") return;
    const listeners = new Map();
    mount.on = (name, callback) => {
      if (!listeners.has(name)) listeners.set(name, []);
      listeners.get(name).push(callback);
      return mount;
    };
    mount.removeAllListeners = name => {
      if (name) listeners.delete(name);
      else listeners.clear();
    };
    mount.__plotly_emit = (name, payload) => {
      for (const callback of listeners.get(name) || []) callback(payload);
    };
  };
  const assign = async (mount, data, layout) => {
    install_events(mount);
    mount.classList.add("js-plotly-plot");
    mount.data = data || [];
    mount.layout = layout || {};
    return mount;
  };
  const nested_assignment = (target, path, value) => {
    const names = String(path).split(".");
    let owner = target;
    for (const name of names.slice(0, -1)) owner = owner[name] ||= {};
    owner[names[names.length - 1]] = value;
  };
  global.Plotly = {
    newPlot: assign,
    react: assign,
    purge(mount) {
      mount.data = [];
      mount.layout = {};
      mount.classList.remove("js-plotly-plot");
      mount.removeAllListeners?.();
    },
    async relayout(mount, update) {
      mount.layout ||= {};
      for (const [path, value] of Object.entries(update || {})) nested_assignment(mount.layout, path, value);
      return mount;
    },
    async restyle(mount, update, indices) {
      const selected = Array.isArray(indices) ? indices : (mount.data || []).map((_trace, index) => index);
      for (const [path, values] of Object.entries(update || {})) {
        selected.forEach((trace_index, value_index) => {
          const value = Array.isArray(values) ? values[value_index] : values;
          nested_assignment(mount.data[trace_index], path, value);
        });
      }
      return mount;
    },
    Plots: {resize() {}},
  };
})(typeof window === "undefined" ? globalThis : window);
// ^^^ THOG
