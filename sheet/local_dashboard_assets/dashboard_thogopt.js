// vvv THOG
/* Optimizer history controls are isolated from the established weight controls. */
(function () {
  "use strict";
  const families = ["attn_q_head_N", "attn_k_head_N", "attn_v_head_N", "attn_out_head_N", "mlp_up", "mlp_down"];
  const quantities = {
    momentum: ["momentum", "raw_momentum", "adaptive_update"],
    scaling: ["second_moment", "raw_second_moment", "rms", "scaling", "adaptive_update"]
  };
  function formatted(value, precision) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return Number(value).toFixed(Math.max(0, Math.min(12, Number(precision) || 0)));
  }
  function aligned_difference(left, right) {
    if (left.length !== right.length || left.some((row, i) => row.length !== right[i].length)) throw new Error("History shapes differ");
    return left.map((row, i) => row.map((value, j) => value - right[i][j]));
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {formatted, aligned_difference, quantities};
  if (typeof document === "undefined") return;
  const controls = {}, payloads = {}, requests = {};
  let identity = "", refresh_timer, inspector_serial = 0;
  function element(tag, text, class_name) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (class_name) node.className = class_name;
    return node;
  }
  function select(options, label) {
    const node = element("select"); node.setAttribute("aria-label", label);
    options.forEach(([value, text]) => { const option = element("option", text); option.value = value; node.append(option); });
    return node;
  }
  function input(label, value, type = "number") {
    const node = element("input"); node.type = type; node.value = value;
    node.setAttribute("aria-label", label); node.title = label; node.style.width = "85px";
    if (type === "number") node.min = "0";
    return node;
  }
  function run_ids() {
    const workspace = window.__instra_workspace;
    if (workspace?.active?.()) return workspace.visible_runs().map(run => run_identifier(run)).filter(Boolean);
    return app.current_run_id ? [app.current_run_id] : [];
  }
  function group(kind) {
    const section = element("section", undefined, "chart-group thogopt-group");
    section.dataset.chartGroup = `optimizer_${kind}`;
    const header = element("header", undefined, "chart-group-header");
    const toggle = element("button", undefined, "chart-group-toggle"); toggle.type = "button";
    toggle.append(element("span", "⌄", "group-caret"), element("strong", kind === "momentum" ? "Momentum history" : "Scaling history"));
    const grid = element("div", undefined, "chart-grid"); grid.id = `optimizer_${kind}_grid`;
    toggle.setAttribute("aria-controls", grid.id); toggle.setAttribute("aria-expanded", "true");
    const quantity = select(quantities[kind].map(value => [value, value.replaceAll("_", " ")]), "History quantity");
    const start = input("First captured step", ""), end = input("Last captured step", "");
    start.placeholder = "First step"; end.placeholder = "Last step";
    const latest = input("Latest captured step per run", "", "checkbox"); latest.checked = true; latest.style.width = "auto";
    const latest_label = element("label", " Latest"); latest_label.prepend(latest);
    const refresh = element("button", "Refresh"); refresh.type = "button";
    const status = element("span", "Waiting for captured optimizer histories."); status.style.fontSize = "12px";
    const toolbar = element("div"); toolbar.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:8px 16px";
    toolbar.append(quantity, start, end, latest_label, refresh, status);
    header.append(toggle); section.append(header, toolbar, grid);
    controls[kind] = {section, quantity, start, end, latest, status};
    for (const family of families) {
      const key = `thogopt_${kind}_${family}`;
      chart_titles[key] = `${kind === "momentum" ? "Momentum" : "Scaling"} · ${family}`;
      app.dynamic_chart_metadata[key] = {x_label: "Executed layer", y_label: quantity.value.replaceAll("_", " ")};
      const card = depth_card(key);
      card.classList.add("thogopt-card");
      const inspect = element("button", "⌕", "thogopt-inspect"); inspect.type = "button";
      inspect.title = "Inspect optimizer history values"; inspect.setAttribute("aria-label", inspect.title);
      inspect.addEventListener("click", () => open_inspector(kind, family));
      card.querySelector(".chart-card-actions").prepend(inspect);
      grid.append(card);
    }
    [quantity, start, end, latest].forEach(node => node.addEventListener("change", () => { if (node === start || node === end) latest.checked = false; refresh_group(kind); }));
    refresh.addEventListener("click", () => refresh_group(kind));
    return section;
  }
  async function request(url) {
    const response = await fetch(url); const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    return payload;
  }
  async function refresh_group(kind) {
    const control = controls[kind], ids = run_ids();
    const serial = requests[kind] = (requests[kind] || 0) + 1;
    if (!ids.length) return;
    control.status.textContent = "Loading optimizer histories…";
    try {
      const results = await Promise.all(ids.map(async id => {
        const query = new URLSearchParams({run: id, quantity: control.quantity.value, latest: String(control.latest.checked)});
        if (control.start.value !== "") query.set("step_min", control.start.value);
        if (control.end.value !== "") query.set("step_max", control.end.value);
        return {id, payload: await request(`/api/optimizer-history?${query}`)};
      }));
      if (serial !== requests[kind]) return;
      payloads[kind] = results;
      for (const family of families) {
        const key = `thogopt_${kind}_${family}`;
        const figure = {data: [], layout: {}};
        const errors = [];
        for (const result of results) {
          const supplied = result.payload.figures[family];
          if (!supplied) continue;
          figure.layout = supplied.layout;
          figure.data.push(...supplied.data.map(trace => ({...trace, name: `${result.id} · ${trace.name}`, meta: {...trace.meta, instra_workspace_run_id: result.id}})));
          errors.push(...(result.payload.errors[family] || []));
        }
        chart_titles[key] = `${control.quantity.value.replaceAll("_", " ")} · ${family}`;
        document.querySelector(`[data-chart="${key}"] h2`).textContent = chart_titles[key];
        app.dynamic_chart_figures[key] = figure;
        app.dynamic_chart_metadata[key].y_label = control.quantity.value.replaceAll("_", " ");
        const placeholder = by_id(`${key}_placeholder`); placeholder.hidden = figure.data.length > 0;
        placeholder.textContent = "No optimizer history captured in this step range.";
        const worst = errors.length ? Math.max(...errors.map(item => item.maximum_absolute_error)) : null;
        by_id(`${key}_detail`).textContent = worst === null ? "Captured histories; reference unavailable" : `Max absolute error ${worst.toExponential(3)} · inspect for RMS and relative L2`;
        if (figure.data.length) await render_plot(by_id(`${key}_plot`), figure, key);
        else if (by_id(`${key}_plot`)) clear_plot(by_id(`${key}_plot`));
      }
      const count = results.reduce((sum, result) => sum + result.payload.steps.length, 0);
      control.status.textContent = `${count} captured steps · × points: AdamW reference · dotted difference: legend toggle`;
    } catch (error) { if (serial === requests[kind]) control.status.textContent = error.message; }
  }
  function open_inspector(kind, family) {
    document.getElementById("thogopt_inspector")?.remove();
    const serial = ++inspector_serial;
    const records = payloads[kind] || [];
    const modal = element("dialog"); modal.id = "thogopt_inspector";
    modal.style.cssText = "width:92vw;height:84vh;max-width:none;padding:16px;border:1px solid #aaa;border-radius:8px;z-index:10000";
    const title = element("h2", `Inspect ${family} · ${controls[kind].quantity.value.replaceAll("_", " ")}`);
    const close = element("button", "Close"); close.onclick = () => { inspector_serial++; modal.close(); modal.remove(); };
    const run = select(records.map(record => [record.id, record.id]), "Source run");
    const reference = select([["", "Same-gradient sampled reference"], ...records.map(record => [record.id, record.id])], "Reference run");
    const step = select([], "Captured step");
    const mode = select([["sample", "Sampled values"], ["difference", "Sampled difference"], ["matrix", "Full matrix"], ["matrix_difference", "Full matrix difference"], ["coefficients", "History coefficients"]], "Inspection view");
    const layer = input("Executed layer (one based)", "1"); layer.min = "1";
    const coefficient = input("Coefficient index (zero based; blank for layer values)", ""); coefficient.placeholder = "Coeff #";
    const precision = input("Decimal places", "4"); precision.max = "12";
    const bar = element("div"); bar.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px";
    bar.append(run, step, mode, reference, element("span", "Layer"), layer, coefficient, element("span", "Decimals"), precision, close);
    const detail = element("p"); const scroll = element("div");
    scroll.style.cssText = "position:relative;overflow:auto;height:calc(100% - 150px);border:1px solid #ddd;font:12px monospace";
    modal.append(title, bar, detail, scroll); document.body.append(modal); modal.showModal();
    const current = () => records.find(record => record.id === run.value)?.payload;
    function steps() {
      const previous = step.value; step.replaceChildren();
      (current()?.steps || []).forEach(value => { const option = element("option", String(value)); option.value = String(value); step.append(option); });
      step.value = [...step.options].some(option => option.value === previous) ? previous : String(current()?.steps?.at(-1) || "");
    }
    let matrix = false, matrix_shape = null, pending = 0, generation = 0;
    function table(values, labels, columns) {
      const table_node = element("table"); table_node.style.borderCollapse = "collapse";
      const head = element("tr"); ["", ...columns].forEach(text => head.append(element("th", text))); table_node.append(head);
      values.forEach((row, index) => { const tr = element("tr"); tr.append(element("th", labels[index])); row.forEach(value => {
        const cell = element("td", formatted(value, precision.value)); cell.style.cssText = "padding:5px 12px;text-align:right;border:1px solid #eee"; cell.title = String(value); tr.append(cell);
      }); table_node.append(tr); });
      scroll.replaceChildren(table_node);
    }
    async function window_values(reset = false) {
      const ticket = ++generation;
      const row_start = reset ? 0 : Math.max(0, Math.floor(scroll.scrollTop / 26) - 1);
      const column_start = reset ? 0 : Math.max(0, Math.floor(scroll.scrollLeft / 110) - 1);
      const query = new URLSearchParams({run: run.value, step: step.value, chart: family, quantity: controls[kind].quantity.value,
        layer: layer.value, row_start, column_start, row_count: 40, column_count: 24});
      if (mode.value === "coefficients" && coefficient.value !== "") query.set("coefficient", coefficient.value);
      if (mode.value === "matrix_difference") {
        if (!reference.value) { detail.textContent = "Choose a captured reference run; sampled same-gradient history is not a full matrix."; return; }
        query.set("reference_run", reference.value);
      }
      try {
        const result = await request(`/api/optimizer-history-matrix?${query}`);
        if (serial !== inspector_serial || ticket !== generation) return;
        matrix_shape = result.shape;
        const canvas = element("div"); canvas.style.cssText = `position:relative;width:${result.shape[1] * 110}px;height:${result.shape[0] * 26}px`;
        result.values.forEach((row, i) => row.forEach((value, j) => {
          const cell = element("div", formatted(value, precision.value));
          cell.style.cssText = `position:absolute;top:${(row_start+i)*26}px;left:${(column_start+j)*110}px;width:110px;height:26px;border:1px solid #eee;text-align:right;padding:4px;box-sizing:border-box`;
          cell.title = `row ${row_start+i}, column ${column_start+j}: ${value}`; canvas.append(cell);
        }));
        scroll.replaceChildren(canvas);
        detail.textContent = `${result.shape.join(" × ")} · step ${result.step}, layer ${result.layer} · ${result.coefficient === null ? "materialised values" : `raw coefficient ${result.coefficient}`} · ${result.comparison || result.kind}; hover for full precision and coordinates`;
      } catch (error) { if (ticket === generation) { scroll.replaceChildren(); detail.textContent = error.message; } }
    }
    async function render() {
      generation++; matrix = ["matrix", "matrix_difference", "coefficients"].includes(mode.value);
      scroll.scrollTop = 0; scroll.scrollLeft = 0;
      if (matrix) {
        if (mode.value === "coefficients" && coefficient.value === "") coefficient.value = "0";
        await window_values(true); return;
      }
      const snapshot = current()?.snapshots.find(item => String(item.optimizer_update) === step.value);
      const field = snapshot?.families[family]; const quantity = controls[kind].quantity.value;
      if (!field) { detail.textContent = "No sampled history at this step."; scroll.replaceChildren(); return; }
      let values = field.values[quantity];
      let other = field.reference?.values[quantity];
      if (reference.value) {
        const other_field = records.find(record => record.id === reference.value)?.payload.snapshots.find(item => String(item.optimizer_update) === step.value)?.families[family];
        if (other_field && JSON.stringify(other_field.coordinates) === JSON.stringify(field.coordinates)) other = other_field.values[quantity];
        else other = null;
      }
      if (mode.value === "difference") {
        if (!other) { detail.textContent = "Matching reference coordinates and step were not captured."; scroll.replaceChildren(); return; }
        try { values = aligned_difference(values, other); } catch (error) { detail.textContent = error.message; return; }
      }
      table(values, field.coordinates.map(pair => `r${pair[0]} c${pair[1]}`), values[0].map((_, i) => `layer ${i+1}`));
      const errors = field.errors?.[quantity];
      detail.textContent = reference.value ? "Separate training runs; difference = source minus reference." : `Same-gradient reference history starts at step ${field.reference?.origin ?? "unavailable"}.`;
      if (errors && !reference.value) detail.textContent += ` Max ${errors.maximum_absolute_error.toExponential(4)}, RMS ${errors.rms_error.toExponential(4)}, relative L2 ${errors.relative_l2_error.toExponential(4)}.`;
      const projected = field.errors?.projected_adaptive_update;
      if (quantity === "adaptive_update" && projected && !reference.value) detail.textContent += ` After matching weight projection: max ${projected.maximum_absolute_error.toExponential(4)}, RMS ${projected.rms_error.toExponential(4)}, relative L2 ${projected.relative_l2_error.toExponential(4)}.`;
    }
    scroll.addEventListener("scroll", () => { if (matrix && matrix_shape) { clearTimeout(pending); pending = setTimeout(() => window_values(), 70); } });
    [step, mode, reference, layer, coefficient, precision].forEach(node => node.addEventListener("change", render));
    run.addEventListener("change", () => { steps(); render(); }); steps(); render();
  }
  function boot() {
    if (typeof app === "undefined" || typeof depth_card !== "function" || !document.getElementById("charts_scroll")) { setTimeout(boot, 100); return; }
    const root = by_id("charts_scroll"); root.append(group("momentum"), group("scaling"));
    const poll = () => {
      if (document.hidden) return;
      const next = JSON.stringify(run_ids());
      if (next !== identity) { identity = next; refresh_group("momentum"); refresh_group("scaling"); }
    };
    setInterval(poll, 1000); poll();
    refresh_timer = setInterval(() => { if (!document.hidden && run_ids().length) { refresh_group("momentum"); refresh_group("scaling"); } }, 30000);
    window.__instra_thogopt = {refresh: () => { refresh_group("momentum"); refresh_group("scaling"); }, open_inspector};
  }
  boot();
})();
// ^^^ THOG
