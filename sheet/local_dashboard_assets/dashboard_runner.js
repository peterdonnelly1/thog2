// vvv THOG render the four Runner views from validated backend state without submitting shell text
"use strict";
(function install_runner() {
  const by_id = name => document.getElementById(name);
  const view = by_id("runner_view");
  if (!view) return;
  const list = by_id("runner_list"), detail = by_id("runner_detail"), message = by_id("runner_message");
  const multiview = by_id("runner_multiview_panel");
  const categories = ["GPT-2 Hyperparameters", "Resume and Fork", "Run Control Parameters", "Geometry", "Width Activation Curves", "Premat", "NSIGHT",
    "Coarse", "Layer Spacing", "Variable Depth", "Chaos Bumps", "Instrumentation"];
  const main_table_order = ["--geometry-preset", "--optimizer", "--n-layer", "DEPTH.order", "WIDTH.order", "--warmup-iters",
    "--block-size", "--n-embd", "--n-head", "--gradient-accumulation-steps", "--checkpoint-segment-size",
    "--learning-rate", "--min-lr", "--max-iters", "--batch-size", "--log-interval", "--eval-iters", "--eval-interval"];
  const category_first_fields = {"Coarse":["--plastic__coarse_phase"],
    "Variable Depth":["--plastic__do_learn_layer_count","--no-plastic__do_learn_layer_count"],
    "Chaos Bumps":["--chaos_bump__sampling__enabled","--no-chaos_bump__sampling__enabled"]};
  const required_parameters = ["--geometry-preset", "--optimizer", "--n-layer", "--warmup-iters", "--block-size",
    "--n-embd", "--n-head", "--gradient-accumulation-steps", "--checkpoint-segment-size",
    "--learning-rate", "--min-lr", "--max-iters", "--batch-size"];
  let snapshot = null, network = null, tab = "recipes", chosen = null, category = "Frequently Used";
  let draft = null, draft_id = null, dirty = false, visible = false, polling = false, last_history_grid = null, last_seen_grid = null;
  let editing_existing = false;
  let current_run_metrics = new Map(), metrics_loading = false;
  let required_notice_seen = false, message_timer = null;
  let manual_grid_selection = false;
  const viewed_files = new Map();
  let download_request = null, download_serial = 0;
  const saved_defaults = () => { try { return JSON.parse(localStorage.getItem("thog2_runner_field_defaults") || "{}"); }
    catch (_) { return {}; } };
  const remember_default = (key, value) => {
    const values = saved_defaults(); values[key] = value;
    localStorage.setItem("thog2_runner_field_defaults", JSON.stringify(values));
  };
  const clear_message = () => { clearTimeout(message_timer); message.textContent = ""; };
  function rename_dialog(title, current) {
    const dialog = by_id("runner_rename_dialog"), form = by_id("runner_rename_form"), input = by_id("runner_rename_input");
    dialog.querySelector("h2").textContent = title;
    input.value = current;
    return new Promise(resolve => {
      const finish = value => {
        form.onsubmit = null; by_id("runner_rename_cancel").onclick = null; dialog.oncancel = null;
        dialog.close(); resolve(value);
      };
      form.onsubmit = event => { event.preventDefault(); finish(input.value.trim()); };
      by_id("runner_rename_cancel").onclick = () => finish(null);
      dialog.oncancel = event => { event.preventDefault(); finish(null); };
      dialog.showModal(); input.focus(); input.select();
    });
  }
  const add = (parent, tag, value, class_name) => {
    const element = document.createElement(tag);
    if (value !== undefined) element.textContent = String(value ?? "—");
    if (class_name) element.className = class_name;
    if(String(value).trim().toLowerCase()==="dense")element.classList.add("instra-dense-preset");
    parent.append(element);
    return element;
  };
  const button = (parent, title, action) => {
    const control = add(parent, "button", title); control.type = "button";
    control.addEventListener("click", action); return control;
  };
  async function request(url, options) {
    const abort = new AbortController(), timeout = setTimeout(() => abort.abort(), 18000);
    try {
      const response = await fetch(url, {...options, signal:abort.signal});
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
      return value;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("Runner request timed out. Check Backend status and retry.");
      throw error;
    } finally { clearTimeout(timeout); }
  }
  async function action(name, values) {
    const quiet = ["delete_recipe","preview","repeat_preview","rename_grid","rename_recipe"].includes(name);
    clear_message();
    if (name === "save" || name === "preview" || name === "launch") required_notice_seen = true;
    message.textContent = name === "stop" ? "attempting to stop grid..." : name === "kill_flush" ?
      "Force stop requested; checking attempts and GPU reservations..." : quiet ? "" : `${name}…`;
    try {
      const result = await request("/api/runner/action", {method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({action:name, ...values})});
      message.textContent = name === "stop" ? "Stop requested; waiting for running attempts to finish..." :
        name === "kill_flush" ? "Kill and Flush started; Grid stays in Progress until GPUs are released." : quiet ? "" : `${name} complete`;
      if (message.textContent) message_timer = setTimeout(clear_message, 10000);
      if (!["preview","repeat_preview"].includes(name)) await refresh(true);
      return result;
    } catch (error) {
      message.textContent = error.message;
      message_timer = setTimeout(clear_message, 18000);
      if (tab === "recipes" && by_id("runner_validation")) by_id("runner_validation").textContent = error.message;
      throw error;
    }
  }
  // vvv THOG WIDTH is derived from the chosen preset and remains visible in Geometry
  const width_presets = new Set(["width", "width-type-I"]);
  function apply_width_selection(recipe) {
    const parameters = recipe.parameters || {};
    const presets = Array.isArray(parameters["--geometry-preset"]) ? parameters["--geometry-preset"] : [parameters["--geometry-preset"]];
    if (presets.some(preset=>width_presets.has(preset))) parameters["--select-width"] = true;
    else delete parameters["--select-width"];
    return recipe;
  }
  // ^^^ THOG
  function current_recipe() {
    if (draft) return apply_width_selection(draft);
    const defaults = saved_defaults();
    const power_caps = {};
    for (const host of network?.hosts || []) if (/^dreedle$/i.test(host.display_name || ""))
      for (const gpu of host.last_discovered?.gpus || [])
        power_caps[gpu.gpu_id || `${host.thog_host_id}.gpu.${gpu.gpu_key}`] = 200;
    for (const [gpu_id,cap] of Object.entries(defaults.power_caps || {})) {
      if (cap == null) delete power_caps[gpu_id];else power_caps[gpu_id]=cap;
    }
    const special = new Set(["profiling_mode","recipe_label","gpu_pool","power_caps"]);
    return apply_width_selection({label:defaults.recipe_label||"New Grid Recipe", parameters:{...{"--max-iters":50,"--batch-size":16,"--geometry-preset":"depth",
      "--n-layer":16,"DEPTH.order":12,"--n-embd":1024,"--n-head":16,"--block-size":1024,
      "--gradient-accumulation-steps":6,"--checkpoint-segment-size":4,"--optimizer":"adamw",
      "--learning-rate":.0009,"--min-lr":.00009,"--warmup-iters":0},...Object.fromEntries(Object.entries(defaults).filter(([key])=>!special.has(key)))},
      power_caps, gpu_pool:defaults.gpu_pool||[],
      profilers:defaults.profiling_mode==="pair"?["nsys","ncu"]:[defaults.profiling_mode||"none"]});
  }
  function recipe_problems(recipe, hosts, require_gpu=true) {
    const errors = [];
    if (!String(recipe.label || "").trim()) errors.push("Recipe label is required");
    const parameters = recipe.parameters || {};
    for (const key of required_parameters) {
      if (parameters[key] === undefined || parameters[key] === null || parameters[key] === "" ||
          Array.isArray(parameters[key]) && !parameters[key].length) errors.push(`${key} is required`);
    }
    const presets = Array.isArray(parameters["--geometry-preset"]) ? parameters["--geometry-preset"] : [parameters["--geometry-preset"]];
    if (presets.length===1 && presets[0]==="dense" || parameters["--model-type"]==="dense")
      for (const key of Object.keys(parameters)) if (snapshot?.catalogue[key]?.dense_compatible===false)
        errors.push(`${key} is not applicable to a dense-only Recipe`);
    // vvv THOG require the selected width axis without implicitly selecting fixed depth
    if (presets.some(preset=>width_presets.has(preset)) && parameters["--model-type"] !== "dense") {
      const width_order = parameters["WIDTH.order"];
      if (width_order === undefined || width_order === null || width_order === "" ||
          Array.isArray(width_order) && !width_order.length) errors.push("WIDTH.order is required for a width-type-I Recipe");
    }
    const depth_options = parameters["--select-depth"];
    const depth_selected = presets.includes("depth") || (Array.isArray(depth_options) ? depth_options.includes(true) : depth_options === true);
    if (depth_selected &&
        parameters["--model-type"] !== "dense" &&
        (parameters["DEPTH.order"] === undefined || parameters["DEPTH.order"] === null || parameters["DEPTH.order"] === "" ||
          Array.isArray(parameters["DEPTH.order"]) && !parameters["DEPTH.order"].length)) {
      errors.push("DEPTH.order is required for a DEPTH Recipe");
    }
    // ^^^ THOG
    if (!Array.isArray(recipe.profilers) || !recipe.profilers.length) errors.push("Profiling mode is required");
    const eligible = (hosts || []).filter(host => host.local || host.execution_enabled)
      .flatMap(host => (host.last_discovered?.execution_profiles?.length ? host.last_discovered.gpus || [] : [])
        .map(gpu => gpu.gpu_id || `${host.thog_host_id}.gpu.${gpu.gpu_key}`));
    if (require_gpu && !eligible.length) errors.push("No discovered host with an execution profile and GPU is eligible; check Networks");
    for (const gpu_id of recipe.gpu_pool || []) {
      if (require_gpu && !eligible.includes(gpu_id)) errors.push(`Selected GPU ${gpu_id} is no longer eligible`);
    }
    for (const key of required_parameters) {
      const values = Array.isArray(parameters[key]) ? parameters[key] : [parameters[key]];
      if (values.some(value => value !== undefined && value !== "" &&
          ["--geometry-preset", "--optimizer"].includes(key) === false && !Number.isFinite(Number(value))))
        errors.push(`${key} must contain valid numbers`);
    }
    const scalar = key => Number(parameters[key]);
    if (Number.isFinite(scalar("--n-embd")) && Number.isFinite(scalar("--n-head")) &&
        scalar("--n-head") > 0 && scalar("--n-embd") % scalar("--n-head") !== 0)
      errors.push("--n-embd must be divisible by --n-head");
    if (Number.isFinite(scalar("--warmup-iters")) && Number.isFinite(scalar("--max-iters")) &&
        scalar("--warmup-iters") >= scalar("--max-iters")) errors.push("--warmup-iters must be less than --max-iters");
    if (depth_selected && parameters["DEPTH.order"] !== undefined && scalar("DEPTH.order") > scalar("--n-layer"))
      errors.push("DEPTH.order must not exceed --n-layer");
    if (parameters["--min-lr"] !== undefined && scalar("--min-lr") > scalar("--learning-rate"))
      errors.push("--min-lr must not exceed --learning-rate");
    return errors;
  }
  function check_recipe(require_gpu=true) {
    const errors = recipe_problems(current_recipe(), network?.hosts, require_gpu);
    // vvv THOG update the visible selector immediately when the preset changes
    const width_field = detail.querySelector('input[data-runner-field="--select-width"]');
    if (width_field) {
      width_field.value = current_recipe().parameters["--select-width"] === true ? "true" : "";
      width_field.readOnly = current_recipe().parameters["--select-width"] === true;
    }
    // ^^^ THOG
    const presets=current_recipe().parameters?.["--geometry-preset"];
    // vvv THOG keep inherited depth defaults editable and visibly inactive for WIDTH alone
    const preset_values=Array.isArray(presets) ? presets : [presets];
    const width_only=preset_values.every(preset=>width_presets.has(preset)) &&
      current_recipe().parameters["--select-depth"] !== true;
    const width_note=by_id("runner_width_selection_note");
    if (width_note) width_note.hidden=!width_only;
    for (const field of detail.querySelectorAll('input[data-runner-field^="DEPTH."]')) {
      field.dataset.inactive=String(width_only);
      field.title=field_help(snapshot.catalogue[field.dataset.runnerField]) +
        (width_only ? " Inactive for width-only: enable --select-depth to combine WIDTH and DEPTH." : "");
    }
    // ^^^ THOG
    const dense_only=(Array.isArray(presets) ? presets.length===1 && presets[0]==="dense" : presets==="dense") ||
      current_recipe().parameters?.["--model-type"]==="dense";
    for (const field of detail.querySelectorAll("input[data-runner-field]")) {
      const key=field.dataset.runnerField,spec=snapshot?.catalogue[key];
      if (spec) field.setAttribute("aria-invalid",String(Boolean(invalid_field_value(key,field.value,spec) ||
        dense_only && spec.dense_compatible===false && field.value.trim())));
    }
    for (const field of detail.querySelectorAll('input[aria-invalid="true"]')) errors.push(`${field.dataset.runnerField} has invalid syntax`);
    const notice = by_id("runner_required_fields");
    if (notice) {
      notice.replaceChildren();
      if (errors.length) add(notice,"strong",`${errors.length} input issue${errors.length === 1 ? "" : "s"}; see details beside Search.`);
      else if (!required_notice_seen) {
        add(notice,"span","Required inputs complete. Host and GPU placement is checked again during Preview.");
        required_notice_seen = true;
      }
    }
    const validation = by_id("runner_validation");
    if (validation) validation.textContent = errors.join("; ");
    return !errors.length;
  }
  function edit(key, value) { clear_message(); draft = current_recipe(); draft.parameters[key] = value; dirty = true; check_recipe(); }
  function parse_value(key, raw, spec) {
    if (spec.kind === "list") return raw.split("\n").filter(Boolean);
    const scalar = value => spec.type === "int" || spec.type === "float" ? Number(value) :
      spec.type === "flag" && ["true","false"].includes(value) ? value === "true" : value;
    return spec.kind === "dimension" ? raw.split(",").map(value => scalar(value.trim())) : scalar(raw);
  }
  function format_duration(value) {
    if (value == null || !Number.isFinite(Number(value))) return "unknown";
    const total = Math.max(0,Math.round(Number(value)));
    const hours = Math.floor(total/3600), minutes = Math.floor(total%3600/60), seconds = total%60;
    return [hours ? `${hours}h` : "", hours || minutes ? `${minutes}m` : "", `${seconds}s`].filter(Boolean).join(" ");
  }
  function grid_elapsed(grid, current_time = Date.now()) {
    const attempts = (grid.runs || []).flatMap(run => (run.attempts || []).map((attempt,index) => ({...attempt, run, is_latest:index===(run.attempts || []).length-1})));
    const starts = attempts.map(attempt => Date.parse(attempt.started_at)).filter(Number.isFinite);
    const start = Date.parse(grid.started_at);
    const first = Number.isFinite(start) ? start : starts.length ? Math.min(...starts) : null;
    if (first === null) return 0;
    let end = Date.parse(grid.finished_at);
    if (!Number.isFinite(end)) {
      if (!["completed", "failed", "cancelled"].includes(grid.state)) end = current_time;
      else {
        const ends = attempts.filter(attempt => attempt.is_latest).map(attempt => {
          const finished = Date.parse(attempt.finished_at);
          if (Number.isFinite(finished)) return finished;
          if (attempt.is_latest) {
            const seconds = attempt.run.duration_seconds;
            if (seconds != null && Number.isFinite(Number(seconds))) return Date.parse(attempt.started_at) + Number(seconds)*1000;
          }
          return NaN;
        });
        if (!ends.length || ends.some(value => !Number.isFinite(value))) return null;
        end = Math.max(...ends);
      }
    }
    return Math.max(0, (end-first)/1000);
  }
  function estimate_range(grid) {
    const interval = grid.estimated_duration?.interval_seconds;
    return Array.isArray(interval) && interval.length === 2 && interval.every(value => value != null && Number.isFinite(Number(value)))
      ? interval.map(format_duration).join(" – ") : "unknown";
  }
  function history_outcome(grid) {
    const completed = (grid.runs || []).filter(run=>run.state === "completed").length;
    return completed && completed === grid.runs.length ? "complete" : completed ? "partial" : "none";
  }
  function field_help(spec) {
    const options = spec.choices?.length ? spec.choices.join(", ") :
      ["flag","bool_value"].includes(spec.type) ? "true, false" :
      spec.type === "int" ? "nonnegative whole numbers" : spec.type === "float" ? "finite numbers" :
      spec.kind === "list" ? "one text value per line" : "text values";
    return `${spec.help || ""}\nAvailable options: ${options}.${spec.kind === "dimension" ? " Use commas for a sweep." : ""}`;
  }
  function invalid_field_value(key, raw, spec) {
    if (!raw.trim()) return required_parameters.includes(key);
    const parts = spec.kind === "dimension" ? raw.split(",").map(part=>part.trim()) :
      spec.kind === "list" ? raw.split("\n") : [raw.trim()];
    if (parts.some(part=>!part)) return true;
    if (spec.kind === "dimension" && (parts.length>64 || new Set(parts).size !== parts.length)) return true;
    return parts.some(part => {
      if (spec.choices?.length && !spec.choices.some(choice=>String(choice)===part)) return true;
      // vvv THOG the width capture window alone admits the documented unbounded sentinel
      if (spec.type === "int") {
        if (key === "--instrumentation__width_activation_curves__end_step" && part === "-1") return false;
        return !/^\+?\d+$/.test(part) || !Number.isSafeInteger(Number(part)) || key === "DEPTH.order" && Number(part)<1;
      }
      // ^^^ THOG
      if (spec.type === "float") return !Number.isFinite(Number(part));
      if (["flag","bool_value"].includes(spec.type)) return !["true","false"].includes(part);
      return part.includes("\0");
    });
  }
  function premat_enabled(recipe) {
    const value = recipe.parameters?.["--premat"];
    return (Array.isArray(value) ? value : [value]).some(option=>option === "enabled" || option === true);
  }
  const enabled_option = value => value === true || value === "true" || value === "enabled";
  function category_enabled(name, recipe) {
    if(name === "Premat")return premat_enabled(recipe);
    if(name === "NSIGHT")return (recipe.profilers || []).some(value=>value !== "none");
    const key={"Coarse":"--plastic__coarse_phase","Variable Depth":"--plastic__do_learn_layer_count",
      "Chaos Bumps":"--chaos_bump__sampling__enabled"}[name];
    if(!key)return false;
    const options=value=>Array.isArray(value) ? value : [value];
    const negative=recipe.parameters?.[key.replace(/^--/,"--no-")];
    return options(recipe.parameters?.[key]).some(enabled_option) &&
      !(negative !== undefined && options(negative).every(enabled_option));
  }
  function update_category_states() {
    for(const control of detail.querySelectorAll(".runner-categories button"))
      control.classList.toggle("runner-category-enabled",category_enabled(control.dataset.runnerCategory,current_recipe()));
  }
  function categories_for_field(key, catalogue, common) {
    return [...(common.includes(key) ? ["Frequently Used"] : []),catalogue[key]?.category].filter(Boolean);
  }
  function matches_search(key, spec, query) {
    const normalize=value=>String(value || "").toLowerCase().replace(/[-_\s]/g,"");
    return normalize(`${key} ${spec.help}`).includes(normalize(query));
  }
  function compare_fields(left, right, name) {
    const order=[...(category_first_fields[name] || []),...main_table_order];
    const rank=key=>{const index=order.indexOf(key);return index<0 ? 999 : index;};
    return rank(left)-rank(right) || left.localeCompare(right);
  }
  function show_fields(container, keys, include_profiler=false) {
    const grid = add(container,"div",undefined,"runner-fields");
    const columns = window.innerWidth < 800 ? 1 : !by_id("runner_parameter_search")?.value &&
      ["Frequently Used","GPT-2 Hyperparameters","Run Control Parameters","Geometry"].includes(category) ? 2 : 1;
    grid.style.setProperty("--runner-columns",String(columns));
    grid.style.setProperty("--runner-rows",String(Math.max(1,Math.ceil((keys.length + (include_profiler ? 1 : 0))/columns))));
    if (include_profiler) {
      const profile_label=add(grid,"label");add(profile_label,"span","Profiling mode");
      profile_label.title="Available options: none, NSYS, NCU, NSYS and NCU pair";
      const profile=add(profile_label,"select");
      for (const [name,value] of [["None","none"],["NSYS","nsys"],["NCU","ncu"],["NSYS and NCU pair","pair"]]) {
        const option=add(profile,"option",name);option.value=value;
      }
      profile.value=current_recipe().profilers?.length===2?"pair":current_recipe().profilers?.[0]||"none";
      profile.addEventListener("change",()=>{draft=current_recipe();draft.profilers=profile.value==="pair"?["nsys","ncu"]:[profile.value];dirty=true;update_category_states();check_recipe();});
      button(profile_label,"Change default value",()=>{remember_default("profiling_mode",profile.value);message.textContent="Profiling default saved";});
    }
    for (const key of keys) {
      const spec = snapshot.catalogue[key];
      if (!spec || spec.ui_hidden || ["manual","automatic"].includes(spec.kind)) continue;
      const label = add(grid,"label");add(label,"span",`${key}${spec.short ? ` (${spec.short})` : ""}`);
      if(by_id("runner_parameter_search")?.value)
        add(label,"small",categories_for_field(key,snapshot.catalogue,snapshot.common).join(" · "),"runner-field-categories");
      label.title = field_help(spec);
      const field = add(label,"input");
      const change_default = button(label,"Change default value",()=>{
        if (field.getAttribute("aria-invalid") === "true") return;
        remember_default(key,field.value === "" ? null : parse_value(key,field.value,spec));
        message.textContent=`Default saved for ${key}`;
      });
      change_default.classList.add("runner-change-default");
      field.dataset.runnerField = key;
      field.title=field_help(spec);
      field.value = Array.isArray(current_recipe().parameters[key]) ? current_recipe().parameters[key].join(spec.kind === "list" ? "\n" : ", ") :
        String(current_recipe().parameters[key] ?? "");
      field.setAttribute("aria-invalid",String(invalid_field_value(key,field.value,spec)));
      field.placeholder = spec.kind === "dimension" ? "One or comma-separated choices" : spec.kind === "list" ? "One item per line" :
        spec.type === "flag" ? "true or false" : String(spec.default ?? "");
      field.addEventListener("input", () => {
        const invalid = invalid_field_value(key,field.value,spec);
        field.setAttribute("aria-invalid",String(invalid));
        if (!invalid) {
          draft=current_recipe();
          if (field.value === "") delete draft.parameters[key];
          else draft.parameters[key]=parse_value(key,field.value,spec);
          dirty=true;
          update_category_states();
        }
        check_recipe();
      });
      field.addEventListener("change", () => {
        if (field.getAttribute("aria-invalid") === "true") return;
        if (field.value === "") { draft=current_recipe();delete draft.parameters[key];dirty=true;check_recipe(); }
        else edit(key, parse_value(key, field.value, spec));
      });
    }
  }
  function render_editor() {
    detail.replaceChildren();
    const recipe = current_recipe();
    add(detail,"h2",draft_id ? `Grid Recipe · ${recipe.label}` : "Grid Recipe");
    const label = add(detail,"label",undefined,"runner-recipe-label");add(label,"span","Recipe label");
    const label_input = add(label,"input"); label_input.value = recipe.label;
    label_input.title="Available options: any label of 1–120 printable characters";
    label_input.dataset.runnerField="Recipe label";
    label_input.addEventListener("input",()=>{
      const value=label_input.value;
      label_input.setAttribute("aria-invalid",String(!value.trim() || value.trim().length>120 || /[\x00-\x1f]/.test(value)));
      draft=current_recipe();draft.label=value;dirty=true;check_recipe();
    });
    label_input.addEventListener("change", () => { draft = current_recipe(); draft.label = label_input.value; dirty = true;check_recipe(); });
    const controls = add(detail,"div",undefined,"runner-actions");
    button(controls,"Save",async () => {
      if (!check_recipe(false)) return;
      try {
        const prior=snapshot.recipes.find(item=>item.recipe_id===draft_id);
        const recipe_id=editing_existing ? draft_id : prior && prior.recipe.label===current_recipe().label ? draft_id : null;
        const saved=await action(editing_existing ? "update_recipe" : "save",{recipe_id,recipe:current_recipe()});
        draft_id=saved.recipe_id;draft=JSON.parse(JSON.stringify(saved.recipe));dirty=false;chosen=draft_id;render();
      }
      catch (_) { /* The error is displayed above. */ }
    });
    // vvv THOG reset only the unsaved editor draft, using the user's saved field defaults
    button(controls,"Reset",() => {
      draft=null;draft_id=null;chosen=null;dirty=false;editing_existing=false;required_notice_seen=false;
      clear_message();render();
    }).title="Reset unsaved configuration to defaults";
    // ^^^ THOG
    button(controls,"Preview",async () => {
      if (!check_recipe()) return;
      try {
        const preview = await action("preview",{recipe:current_recipe()});
        const panel = by_id("runner_preview");
        panel.replaceChildren();
        add(panel,"h3",`Total Runs in Grid: ${preview.total_runs} · physical executions: ${preview.runs.length}`);
        add(panel,"p", preview.estimated_duration.seconds === null ?
          `Estimated Time to complete this grid: ${preview.estimated_duration.explanation}` :
          `Estimated Time to complete this grid: ${format_duration(preview.estimated_duration.seconds)} (${preview.estimated_duration.interval_seconds.map(format_duration).join(" – ")}), ${preview.estimated_duration.confidence} confidence, ${preview.estimated_duration.exemplars} exemplars`,"runner-preview-estimate");
        const placement=add(panel,"div",undefined,"runner-preview-placement");
        for(const place of preview.gpu_pool)add(placement,"p",`Host: ${place.host_label} GPU: ${place.gpu.ordinal} currently free VRAM: ${place.gpu.free_mib??"unknown"}MiB reservation: ${place.reservation_owner?.grid_id||"none"}`);
        run_headings(panel);
        for (const run of preview.runs) render_run(panel,run,true);
      } catch (_) { /* The error is displayed above. */ }
    });
    button(controls,"Launch",async () => {
      if (!check_recipe()) return;
      try {
        const preview = await action("preview",{recipe:current_recipe()});
        const names = preview.gpu_pool.map(item => `${item.host_label} GPU ${item.gpu.ordinal}`).join(", ");
        const power=preview.runs.map(run=>`${run.host_label} GPU ${run.gpu.ordinal}: current ${run.current_power_w??"unknown"} W; default ${run.default_power_w??"unknown"} W; requested ${run.requested_power_w??"default"} W`).filter((value,index,self)=>self.indexOf(value)===index).join("\n");
        const needs_save=!draft_id || dirty;
        if (!confirm(`${needs_save?"Save and Launch":"Launch"} ${preview.total_runs} THOG runs on ${names}?\n\nGPU power limits:\n${power}`)) return;
        if (preview.total_runs > 100 && !confirm(`These values will take the size of the grid to ${preview.total_runs.toLocaleString()} runs. Proceed?`)) return;
        if (needs_save) {
          const prior=snapshot.recipes.find(item=>item.recipe_id===draft_id);
          const recipe_id=editing_existing ? draft_id : prior && prior.recipe.label===current_recipe().label ? draft_id : null;
          const saved=await action(editing_existing ? "update_recipe" : "save",{recipe_id,recipe:current_recipe()});
          draft_id=saved.recipe_id;draft=JSON.parse(JSON.stringify(saved.recipe));dirty=false;
        }
        const launched=await action("launch",{recipe_id:draft_id,confirm_large:preview.total_runs>100});
        last_history_grid=launched.grid_id;chosen=launched.grid_id;manual_grid_selection=true;tab="progress";render();
      } catch (_) { /* The error is displayed above. */ }
    });
    // vvv THOG state the active axes beside the selected Recipe
    const width_note=add(detail,"p","Width only: inherited DEPTH fields are inactive. Enable --select-depth in Geometry to combine both axes.","runner-gpu-note");
    width_note.id="runner_width_selection_note";
    // ^^^ THOG
    const eligible=(network?.hosts||[]).filter(host=>(host.local||host.execution_enabled) &&
      host.last_discovered?.execution_profiles?.length)
      .flatMap(host=>(host.last_discovered?.gpus||[]).map(gpu=>({host,gpu})));
    const placement_text = selected => selected.length ? selected.map(id => {
      const choice=eligible.find(({host,gpu})=>(gpu.gpu_id||`${host.thog_host_id}.gpu.${gpu.gpu_key}`)===id);
      return choice ? `${choice.host.display_name} GPU ${choice.gpu.ordinal} (${choice.gpu.model}, ${choice.gpu.memory_mib??"?"} MiB)` : id;
    }).join("; ") : `Automatic (default) from ${eligible.length} discovered GPU${eligible.length===1?"":"s"}`;
    const placement=add(detail,"p",`GPU placement: ${placement_text(recipe.gpu_pool||[])}` ,"runner-placement-status");
    const pool = add(detail,"details",undefined,"runner-gpu-pool");
    add(pool,"summary",`Eligible hosts and GPUs · ${placement_text(recipe.gpu_pool||[])}`);
    add(pool,"p","Select GPUs to restrict placement. With none selected, Runner chooses from eligible hosts.","runner-gpu-note");
    for (const host of network?.hosts || []) {
      if ((!host.local && !host.execution_enabled) || !host.last_discovered?.execution_profiles?.length) continue;
      for (const gpu of host.last_discovered?.gpus || []) {
        const gpu_id=gpu.gpu_id||`${host.thog_host_id}.gpu.${gpu.gpu_key}`;
        const line=add(pool,"div",undefined,"runner-gpu-row");
        const tick=add(line,"input");tick.type="checkbox";tick.setAttribute("aria-label",`Select ${host.display_name} GPU ${gpu.ordinal}`);
        tick.title="Available options: selected or unselected. With all GPUs unselected, placement is automatic.";
        tick.checked=(recipe.gpu_pool||[]).includes(gpu_id);
        tick.addEventListener("change",()=>{draft=current_recipe();draft.gpu_pool=draft.gpu_pool||[];
          draft.gpu_pool=tick.checked?[...draft.gpu_pool,gpu_id]:draft.gpu_pool.filter(id=>id!==gpu_id);dirty=true;
          pool.querySelector("summary").textContent=`Eligible hosts and GPUs · ${placement_text(draft.gpu_pool)}`;
          placement.textContent=`GPU placement: ${draft.gpu_pool.length ? placement_text(draft.gpu_pool) : placement_text([])}`;check_recipe();});
        add(line,"span",`${host.display_name} · GPU ${gpu.ordinal} · ${gpu.model} · ${gpu.memory_mib??"?"} MiB`);
        const watts=add(line,"label","Power cap (W)");
        watts.title="Available options: blank for the GPU default policy, or integer watts from 50 to 600 within this GPU's supported range. An explicit cap requires driver support.";
        const cap=add(watts,"input");cap.type="number";cap.min="50";cap.max="600";cap.value=recipe.power_caps?.[gpu_id]??"";
        cap.title=watts.title;
        cap.dataset.runnerField=`${host.display_name} GPU ${gpu.ordinal} power cap`;
        cap.addEventListener("input",()=>{
          cap.setAttribute("aria-invalid",String(cap.validity.badInput || Boolean(cap.value) && (!Number.isInteger(Number(cap.value)) || Number(cap.value)<50 || Number(cap.value)>600)));
          check_recipe();
        });
        cap.addEventListener("change",()=>{
          if(cap.getAttribute("aria-invalid")==="true")return;draft=current_recipe();draft.power_caps=draft.power_caps||{};
          if(cap.value)draft.power_caps[gpu_id]=Number(cap.value);else delete draft.power_caps[gpu_id];dirty=true;});
        button(line,"Change default value",()=>{
          const values=saved_defaults().power_caps||{};
          values[gpu_id]=cap.value?Number(cap.value):null;
          remember_default("power_caps",values);message.textContent=`Power default saved for ${host.display_name} GPU ${gpu.ordinal}`;
        }).classList.add("runner-change-default");
      }
    }
    const parameter_panel=add(detail,"section",undefined,"runner-parameter-panel");
    add(parameter_panel,"h3","Run parameters");
    const search_row=add(parameter_panel,"div",undefined,"runner-search-row");
    const search=add(search_row,"input",undefined,"runner-parameter-search");search.id="runner_parameter_search";
    search.placeholder="Search fields across all parameter tabs";
    search.title="Available options: parameter names or help text; leave blank to show the selected category";
    const validation=add(search_row,"p","","runner-validation");validation.id="runner_validation";
    const categories_row=add(parameter_panel,"nav",undefined,"runner-categories");
    const fields=add(parameter_panel,"div");
    const preview_panel=add(detail,"section",undefined,"runner-preview");preview_panel.id="runner_preview";
    function show_category(query="",filter_category=false) {
      fields.replaceChildren();categories_row.replaceChildren();
      preview_panel.hidden=category!=="Frequently Used"||Boolean(query);
      const matching=Object.keys(snapshot.catalogue).filter(key=>!snapshot.catalogue[key].ui_hidden &&
        !["manual","automatic"].includes(snapshot.catalogue[key].kind) && matches_search(key,snapshot.catalogue[key],query));
      const matching_categories=new Set(matching.flatMap(key=>categories_for_field(key,snapshot.catalogue,snapshot.common)));
      const profiling_match=query && matches_search("profiling mode",{help:"none NSYS NCU NSYS and NCU pair"},query);
      if(profiling_match)matching_categories.add("NSIGHT");
      for (const name of ["Frequently Used",...categories]) {
        if(query && !matching_categories.has(name))continue;
        const control=button(categories_row,name,()=>{category=name;show_category(search.value,Boolean(search.value));});
        control.dataset.runnerCategory=name;
        control.classList.toggle("active",category===name && (!query || filter_category));
      }
      const keys=(query?matching.filter(key=>!filter_category || categories_for_field(key,snapshot.catalogue,snapshot.common).includes(category)):
        category==="Frequently Used"?snapshot.common:Object.keys(snapshot.catalogue).filter(key=>snapshot.catalogue[key].category===category))
        .filter(key=>!snapshot.catalogue[key].ui_hidden && !["manual","automatic"].includes(snapshot.catalogue[key].kind))
        .sort((a,b)=>compare_fields(a,b,category));
      show_fields(fields,keys,query ? profiling_match && (!filter_category || category==="NSIGHT") : category==="NSIGHT");
      if(query && !keys.length && !profiling_match)add(fields,"p","No matching fields","runner-search-empty");
      update_category_states();
    }
    search.addEventListener("input",()=>show_category(search.value));show_category();
    check_recipe();
  }
  function wall_time(value) {
    if (!value) return "—";
    const date = new Date(value);
    // vvv THOG local YY-MM-DD HH:mm:ss is explicit and independent of locale AM/PM preferences
    if (!Number.isFinite(date.getTime())) return "—";
    const two = number => String(number).padStart(2,"0");
    return `${two(date.getFullYear()%100)}-${two(date.getMonth()+1)}-${two(date.getDate())}  ${two(date.getHours())}:${two(date.getMinutes())}:${two(date.getSeconds())}`;
    // ^^^ THOG
  }
  // vvv THOG show a per-run finish estimate using live progress, then the launch-time exemplar estimate
  function estimated_run_end(run,observed,current_time=Date.now()) {
    if(run.state!=="running")return null;
    const start=Date.parse(run.attempts?.at(-1)?.started_at || run.started_at || "");
    if(!Number.isFinite(start))return null;
    const total=Number(run.parameters?.["--max-iters"]),step=Number(observed?.maximum_update);
    if((run.attempts?.length || 0)<=1 && Number.isFinite(total) && total>0 && Number.isFinite(step) && step>0 && step<total)
      return new Date(current_time+(current_time-start)*(total-step)/step).toISOString();
    const seconds=run.estimated_duration_seconds;
    return seconds!=null && Number.isFinite(Number(seconds)) && Number(seconds)>0 && start+Number(seconds)*1000>current_time
      ? new Date(start+Number(seconds)*1000).toISOString() : null;
  }
  // ^^^ THOG
  function run_headings(parent) {
    const headings=add(parent,"div",undefined,"runner-run-headings");
    for(const name of ["Run ID","preset","start","end",...(tab==="progress" ? ["est. end"] : []),"Host","GPU","State","Step","Loss","Best loss",...(tab!=="progress" ? ["Profiling"] : [])])add(headings,"strong",name);
  }
  const loss_text = value => value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(3);
  function render_run(parent,run,preview=false) {
    const row=add(parent,"details",undefined,"runner-run");
    row.dataset.runId=run.run_id;
    if (!preview && detail.dataset.gridId===run.grid_id && detail.dataset.openRunIds?.split(",").includes(run.run_id)) row.open=true;
    if (tab === "history" && ["failed","blocked"].includes(run.state)) row.open = true;
    const summary=add(row,"summary",undefined,"runner-run-identity");
    add(summary,"span",run.run_id.slice(0,8));
    const preset=add(summary,"span",run.parameters?.["--geometry-preset"] || run.parameters?.["--model-type"] || "—");
    preset.classList.toggle("instra-dense-preset",preset.textContent.toLowerCase()==="dense");
    const run_start = run.started_at || run.attempts?.[0]?.started_at;
    const run_end = ["completed","failed","cancelled"].includes(run.state) ? run.finished_at || run.attempts?.at(-1)?.finished_at : null;
    add(summary,"span",preview?"—":wall_time(run_start)).title = preview ? "" : (run_start ? new Date(run_start).toLocaleString() : "Start time unavailable");                                                               // <<< THOG show durable run-level start wall time
    add(summary,"span",preview?"—":wall_time(run_end)).title = preview ? "" : (run_end ? new Date(run_end).toLocaleString() : "End time unavailable");                                                         // <<< THOG show durable run-level end wall time
    if(tab==="progress") {
      const estimated=preview ? null : estimated_run_end(run,current_run_metrics.get(run.run_id));
      add(summary,"span",wall_time(estimated),"runner-estimated-end").title=estimated ? "Estimated from current progress or comparable completed runs" : "Estimate unavailable";
    }
    add(summary,"span",run.host_label);
    add(summary,"span",`GPU ${run.gpu.ordinal}`);
    add(summary,"span",run.state,`runner-status runner-status-${run.state}`);
    const observed=current_run_metrics.get(run.run_id);
    add(summary,"span",preview?"—":observed?.maximum_update??"—");
    add(summary,"span",preview?"—":loss_text(observed?.last_loss));
    add(summary,"span",preview?"—":loss_text(observed?.best_loss));
    if(tab!=="progress") {
      const profiler=add(summary,"span",run.profiler==="none"?"No profiling":run.profiler.toUpperCase());
      profiler.title="Profiling mode: none, NSYS or NCU";
    }
    add(row,"p",`GPU ${run.gpu.model} · ${run.gpu.uuid||run.gpu.gpu_key} · ${run.execution_profile} · ${run.dtype}/${run.attention_backend} · peak ${run.required_mib} MiB · power current ${run.current_power_w??"unknown"} W, default ${run.default_power_w??"unknown"} W, requested ${run.requested_power_w??"default"} W, observed ${run.attempts?.at(-1)?.observed_power_w??"pending"} W`);
    if (run.pairing_id) add(row,"p",`Paired profiler runs: ${run.pairing_id}`);
    if (run.blocking_reason) add(row,"p",run.blocking_reason);
    if (run.attempts?.length) {
      const started=Date.parse(run.attempts.at(-1).started_at);
      const seconds=run.duration_seconds??Math.max(0,Math.round((Date.now()-started)/1000));
      add(row,"p",`Elapsed ${format_duration(seconds)} · attempts ${run.attempts.length}`);
    }
    if (run.state==="running"||run.state==="completed"||run.state==="failed") {
      button(row,"Open in Runs",async()=>{
        try {
          const catalogue=await request("/api/runs");
          const item=catalogue.runs.find(entry=>entry.runner_run_id===run.run_id);
          if(item)window.location.href=`/runs/${encodeURIComponent(item.dashboard_run_id)}`;
          else message.textContent="Run not yet in this Runs catalogue; inspect its producing host or enable Monitoring";
        } catch(error){message.textContent=error.message;}
      });
    }
    if (run.state==="failed")button(row,"Retry run",async()=>{
      if(!confirm(`Retry ${run.run_id.slice(0,8)} as a new attempt?`))return;
      try{await action("retry_run",{grid_id:run.grid_id,run_id:run.run_id});tab="progress";render();}
      catch(_){/* Error shown above. */}
    });
    for (const attempt of run.attempts||[]) {
      add(row,"p",`Attempt ${attempt.attempt_id} · ${attempt.started_at} · ${attempt.state||run.state} · exit ${attempt.exit_code??"pending"}`);
      const excerpt=attempt.failure_excerpt ? add(row,"pre",attempt.failure_excerpt,"runner-failure-log") : null;
      if (attempt.state==="failed" || run.state==="failed") {
        const log=add(row,"pre","","runner-failure-log");log.hidden=true;
        button(row,"View attempt log",async()=>{
          try {
            const query=new URLSearchParams({grid_id:run.grid_id,run_id:run.run_id,attempt_id:attempt.attempt_id});
            const result=await request(`/api/runner/log?${query}`);
            log.textContent=result.text||"No log output";log.hidden=false;
            if(excerpt)excerpt.hidden=true;
          } catch(error){log.textContent=`Unable to retrieve attempt log: ${error.message}`;log.hidden=false;message.textContent=error.message;}
        });
      }
    }
    add(row,"pre",JSON.stringify(run.parameters,null,2));
  }
  function render_downloads(parent,grid) {
    const files = [["script", "Runner Script", `${grid.grid_tag}.sh`],
      ["classic", "Old-school THOG script", `${grid.grid_tag}_grid_bash_runner_script.sh`],
      ["manifest", "Resolved Manifest (JSON)", `${grid.grid_tag}_manifest.json`],
      ["placement", "Placement (JSON)", `${grid.grid_tag}_placement.json`],
      ["status", "Status (JSON)", `${grid.grid_tag}_status.json`],
      ...(grid.conversion ? [["conversion", "Conversion (JSON)", `${grid.grid_tag}_conversion.json`]] : []),
      ["log", "Grid event log", `${grid.grid_tag}_events.jsonl`]];
    const controls=add(parent,"div",undefined,"runner-downloads");
    const display=add(parent,"p","", "runner-file-displayed");
    display.setAttribute("role","status");
    const viewer=add(parent,"pre","","runner-script-viewer");
    const view_controls=new Map();
    async function show_file(name) {
      const record=files.find(file=>file[0]===name);
      if(!record)return;
      viewed_files.set(grid.grid_id,name);
      download_request?.abort();download_request=new AbortController();
      const serial=++download_serial;
      for(const [key,control] of view_controls) {control.classList.toggle("active",key===name);control.setAttribute("aria-pressed",String(key===name));}
      display.textContent=`Displaying: ${record[1]} · ${record[2]}`;
      viewer.textContent="Loading…";
      try {
        const response=await fetch(`/api/runner/file?grid_id=${encodeURIComponent(grid.grid_id)}&name=${name}`,{signal:download_request.signal});
        if(!response.ok)throw new Error(`HTTP ${response.status}`);
        const source=await response.text();
        if(serial===download_serial && detail.querySelector(".runner-script-viewer")===viewer) {
          viewer.textContent=source || "Empty file";viewer.scrollTop=0;
        }
      } catch(error) {
        if(error.name!=="AbortError" && serial===download_serial && detail.querySelector(".runner-script-viewer")===viewer)
          viewer.textContent=`${record[1]} unavailable: ${error.message}`;
      }
    }
    for(const [name,label,filename] of files) {
      const row=add(controls,"div",undefined,"runner-download-row");
      add(row,"span",label);
      const control=button(row,"View",()=>show_file(name));
      control.title=`View ${filename}`;control.setAttribute("aria-label",control.title);
      view_controls.set(name,control);
      const link=add(row,"a","Download");
      link.href=`/api/runner/file?grid_id=${encodeURIComponent(grid.grid_id)}&name=${name}&download=1`;
      link.download=filename;link.title=`Download ${filename}`;
    }
    show_file(viewed_files.get(grid.grid_id)||"script");
  }
  function update_grid_state(state,grid) {
    if(!state)return;
    state.replaceChildren();
    add(state,"span","State ");add(state,"span",grid.state,`runner-status runner-status-${grid.state}`);
    add(state,"span",` · ${grid.runs.filter(run=>run.state==="completed").length}/${grid.runs.length} runs completed · Recipe ${grid.recipe_id} · created ${new Date(grid.created_at).toLocaleString()}`);
  }
  function render_grid(grid) {
    const previous_grid=detail.dataset.gridId;
    const configuration_open=previous_grid===grid.grid_id && detail.querySelector(".runner-grid-configuration")?.open;
    const open_runs=previous_grid===grid.grid_id ? [...detail.querySelectorAll(".runner-run[open]")].map(row=>row.dataset.runId) : [];
    const download_signature=JSON.stringify([grid.grid_id,grid.label,Boolean(grid.conversion)]);
    if(tab==="files" && detail.dataset.downloadSignature===download_signature && detail.querySelector(".runner-script-viewer")) {
      update_grid_state(detail.querySelector(".runner-grid-state"),grid);return;
    }
    download_request?.abort();download_serial++;
    detail.replaceChildren();
    detail.dataset.downloadSignature=tab==="files"?download_signature:"";
    detail.dataset.gridId=grid.grid_id;
    detail.dataset.openRunIds=open_runs.join(",");
    const heading = add(detail,"div",undefined,"runner-grid-heading");
    add(heading,"h2",`${grid.grid_tag} · ${grid.label}`);
    if(tab==="progress")button(heading,"Rename",async()=>{
      const label=await rename_dialog(`Rename ${grid.grid_tag}`,grid.label);
      if(label===null || label.trim()===grid.label)return;
      try{await action("rename_grid",{grid_id:grid.grid_id,label});render();}catch(_){/* Error shown above. */}
    });
    const state=add(detail,"p",undefined,"runner-grid-state");
    update_grid_state(state,grid);
    if(tab==="progress" || tab==="history") {
      // vvv THOG retain launched facts separately from readbacks and execution changes
      const configuration=add(detail,"details",undefined,"runner-grid-configuration");
      configuration.open=Boolean(configuration_open);
      add(configuration,"summary","Grid Configuration");
      add(configuration,"h3","Recipe as launched");
      add(configuration,"pre",JSON.stringify(grid.launch_configuration?.recipe||grid.recipe,null,2));
      add(configuration,"h3","Resolved runs and launch-time placement");
      for(const run of grid.runs) {
        const launched=grid.launch_configuration?.runs?.find(item=>item.run_id===run.run_id)||run;
        const record=add(configuration,"details");
        add(record,"summary",`${run.run_id.slice(0,8)} · ${run.host_label} GPU ${run.gpu.ordinal} · ${run.profiler}`);
        add(record,"pre",JSON.stringify(launched,null,2));
        add(record,"h4","Runtime observations and changes");
        add(record,"pre",JSON.stringify({current_host_id:run.host_id,current_gpu:run.gpu,
          state:run.state,blocking_reason:run.blocking_reason||null,
          attempts:run.attempts,released:run.released??false,duration_seconds:run.duration_seconds??null,
          last_release_error:run.last_release_error??null},null,2));
      }
      if(grid.conversion)add(configuration,"pre",JSON.stringify({placement_change:grid.conversion},null,2));
      const manifest=add(configuration,"a","View JSON manifest");
      manifest.href=`/api/runner/file?grid_id=${encodeURIComponent(grid.grid_id)}&name=manifest`;
      manifest.target="_blank";manifest.rel="noopener";
      const events=add(configuration,"a"," · View dated Grid events");
      events.href=`/api/runner/file?grid_id=${encodeURIComponent(grid.grid_id)}&name=log`;
      events.target="_blank";events.rel="noopener";
      // ^^^ THOG
    }
    if(["progress"].includes(tab)) add(detail,"p",
      "Possible states are: ready, queued, dispatching, running, blocked, stopping, flushing, unknown, completed, failed, cancelled.","runner-state-legend");
    if(tab==="progress") {
      const controls=add(detail,"div",undefined,"runner-actions");
      button(controls,"Stop Grid",async()=>{if(!confirm(`Stop ${grid.grid_tag} and its running attempts?`))return;
        try{await action("stop",{grid_id:grid.grid_id});}catch(_){/* Error shown above. */}});
      if(grid.repeat_mode==="tight" && grid.runs.some(run=>["blocked","queued"].includes(run.state))){
        button(controls,"Convert remaining work to Loose",async()=>{
          try{
            const proposed=await action("repeat_preview",{grid_id:grid.grid_id,mode:"loose"});
            const changes=proposed.changes.map(item=>`${item.prior_run_id.slice(0,8)} ${JSON.stringify(item.changes)}`).join("\n");
            if(!confirm(`Convert remaining ${grid.grid_tag} work to Loose placement?\nCompleted and running attempts keep their placement.\n${changes}`))return;
            await action("convert_to_loose",{grid_id:grid.grid_id});
          }catch(_){/* Error shown above. */}
        });
        button(controls,"Fail Grid",async()=>{if(!confirm(`Fail ${grid.grid_tag} and stop active attempts? Completed results remain.`))return;
          try{await action("fail_grid",{grid_id:grid.grid_id});}catch(_){/* Error shown above. */}});
      }
    }
    if (tab==="history") {
      const failures=grid.runs.filter(run=>run.state==="failed");
      if (failures.length) add(detail,"p",`${failures.length} failed run${failures.length===1?"":"s"}. Expand a run for its exit code, failure log and attempts.`,"runner-failure-summary");
      const actions=add(detail,"div",undefined,"runner-actions");
      for (const mode of ["loose","tight"]) button(actions,`Repeat (${mode})`,async()=>{
        try {
          const proposed=await action("repeat_preview",{grid_id:grid.grid_id,mode});
          const places=proposed.gpu_pool.map(p=>`${p.host_label} GPU ${p.gpu.ordinal}`).join(", ");
          const differences=proposed.changes.slice(0,8).map(item=>`${item.prior_run_id.slice(0,8)}: ${JSON.stringify(item.changes)}`).join("\n");
          const estimate=proposed.estimated_duration.seconds===null?"insufficient history":`${format_duration(proposed.estimated_duration.seconds)}`;
          if(!confirm(`Repeat ${grid.grid_tag} (${mode})?\nRecipe: ${proposed.recipe.label}\nRuns: ${proposed.total_runs}; executions: ${proposed.physical_executions}\nPlacement: ${places}\nEstimate: ${estimate}\nChanges: ${differences||"none"}${proposed.changes.length>8?"\nMore changes appear in the resolved preview":""}`))return;
          if(proposed.total_runs>100 && !confirm(`These values will take the size of the grid to ${proposed.total_runs.toLocaleString()} runs. Proceed?`))return;
          const repeated=await action("repeat",{grid_id:grid.grid_id,mode,confirm_large:proposed.total_runs>100});
          last_history_grid=repeated.grid_id;tab="progress";render();
        }
        catch (_) { /* Error shown above. */ }
      });
    }
    if (tab==="files") render_downloads(detail,grid);
    if(tab==="log") {
      const path=`/api/runner/file?grid_id=${encodeURIComponent(grid.grid_id)}&name=log`;
      const download=add(detail,"a","Download Grid event log");download.href=path+"&download=1";download.download="";
      const viewer=add(detail,"pre","Loading Grid events…","runner-event-log");
      viewer.dataset.gridId=grid.grid_id;
      update_event_log();
    }
    if(tab==="history" || tab==="progress") run_headings(detail);
    for (const run of (tab==="history"?grid.runs.map((item,index)=>({item,index})).sort((a,b)=>
      String(b.item.attempts?.at(-1)?.started_at||grid.created_at).localeCompare(
        String(a.item.attempts?.at(-1)?.started_at||grid.created_at)) || b.index-a.index).map(entry=>entry.item):grid.runs))
      if(!["files","log","multiview"].includes(tab))render_run(detail,run);
  }
  async function update_event_log() {
    const viewer=detail.querySelector(".runner-event-log");
    if(tab!=="log" || !viewer || viewer.dataset.gridId!==chosen)return;
    const grid_id=viewer.dataset.gridId;
    try {
      const path=`/api/runner/file?grid_id=${encodeURIComponent(grid_id)}&name=log`;
      const response=await fetch(path,{cache:"no-store"});
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const source=await response.text();
      if(tab!=="log" || chosen!==grid_id || detail.querySelector(".runner-event-log")!==viewer)return;
      const at_bottom=viewer.scrollTop+viewer.clientHeight>=viewer.scrollHeight-24;
      viewer.textContent=source.trim().split("\n").filter(Boolean)
          .map(line=>{try{const event=JSON.parse(line);return `${event.time}  ${event.event.toUpperCase()}  ${event.detail}`;}
            catch(_){return line;}}).join("\n")||"No events yet";
      if(at_bottom)viewer.scrollTop=viewer.scrollHeight;
    }catch(error){if(detail.querySelector(".runner-event-log")===viewer)viewer.textContent=`Grid event log unavailable: ${error.message}`;}
  }
  function show_multiview_selector(grids, selected_id) {
    let control=multiview.querySelector(".runner-multiview-selector");
    if(!control) {
      control=add(multiview,"label","Active Grid · ","runner-multiview-selector");
      const select=add(control,"select");
      select.setAttribute("aria-label","Choose an active Grid for Multiview");
      select.addEventListener("change",()=>{chosen=select.value;manual_grid_selection=Boolean(chosen);render();});
    }
    const select=control.querySelector("select");
    const signature=grids.map(grid=>`${grid.grid_id}|${grid.label}|${grid.state}`).join(";");
    if(control.dataset.options!==signature) {
      select.replaceChildren();
      if(!grids.some(grid=>grid.grid_id===selected_id)) {
        const placeholder=add(select,"option","Select an active Grid");placeholder.value="";
      }
      for(const grid of [...grids].reverse()) {
        const locations=[...new Set(grid.runs.map(run=>`${run.host_label} GPU ${run.gpu.ordinal}`))].join(", ");
        const option=add(select,"option",`${grid.grid_tag} · ${grid.label} · ${grid.state} · ${locations}`);
        option.value=grid.grid_id;
      }
      control.dataset.options=signature;
    }
    select.value=grids.some(grid=>grid.grid_id===selected_id)?selected_id:"";
  }
  function render() {
    if (!snapshot || !visible) return;
    view.classList.toggle("runner-progress",tab==="progress");
    view.classList.toggle("runner-recipes",tab==="recipes");
    for (const control of by_id("runner_tabs").querySelectorAll("button"))control.classList.toggle("active",control.dataset.runnerTab===tab);
    list.replaceChildren();
    const grids=snapshot.grids.filter(grid=>!["progress","multiview"].includes(tab)||!["completed","failed","cancelled"].includes(grid.state));
    if(tab==="recipes") {
      const add_row=add(list,"div",undefined,"runner-recipe-add-row");
      button(add_row,"Add Grid Recipe",()=>{
        if(dirty && !confirm("Discard unsaved Recipe edits and start a new Recipe?"))return;
        clear_message();required_notice_seen=true;
        draft_id=null;draft=null;chosen=null;dirty=false;editing_existing=false;render_editor();
      }).classList.add("runner-add-recipe");
      add(add_row,"p","","runner-required-fields").id="runner_required_fields";
      for(const saved of [...snapshot.recipes].sort((a,b)=>String(b.created_at||b.updated_at||"").localeCompare(String(a.created_at||a.updated_at||"")))) {
        const row=add(list,"div",undefined,"runner-recipe-row");
        const recipe_button=button(row,saved.recipe.label,()=>{if(dirty && !confirm("Discard unsaved Recipe edits?"))return;editing_existing=false;draft_id=saved.recipe_id;chosen=draft_id;
          draft=JSON.parse(JSON.stringify(saved.recipe));dirty=false;render();});
        recipe_button.title=saved.recipe.label;
        recipe_button.classList.toggle("active",chosen===saved.recipe_id);
        add(row,"span",saved.state || "ready","runner-status");
        // vvv THOG explicit Edit selects in-place saving while the name button retains copy saving
        const edit_button=button(row,"Edit",()=>{
          if(dirty && !confirm("Discard unsaved Recipe edits?"))return;
          editing_existing=true;draft_id=saved.recipe_id;chosen=draft_id;
          draft=JSON.parse(JSON.stringify(saved.recipe));dirty=false;render();
        });
        edit_button.classList.add("runner-edit-recipe");edit_button.title="Edit this saved Recipe in place";
        // ^^^ THOG
        const rename=button(row,"Rename",async()=>{
          const label=await rename_dialog("Rename Grid Recipe",saved.recipe.label);
          if(label===null || label.trim()===saved.recipe.label)return;
          if(!label.trim()){message.textContent="Grid Recipe name cannot be blank";return;}
          try{
            await action("rename_recipe",{recipe_id:saved.recipe_id,label:label.trim()});
            if(draft_id===saved.recipe_id && draft)draft.label=label.trim();
            render();
          }catch(_){/* Error shown above. */}
        });
        rename.title="Rename this saved Grid Recipe. To rename a launched Grid, open it in History or Progress.";
        button(row,"Delete",async()=>{
          if(!confirm(`Delete Recipe ${saved.recipe.label}? Its Grid history will remain available.`))return;
          try{await action("delete_recipe",{recipe_id:saved.recipe_id});if(chosen===saved.recipe_id){chosen=null;draft_id=null;draft=null;dirty=false;editing_existing=false;}render();}
          catch(_){/* Error shown above. */}
        }).classList.add("runner-recipe-delete");
      }
      if(chosen && !snapshot.recipes.some(item=>item.recipe_id===chosen))chosen=null;
      if(!dirty || !detail.querySelector("input:focus"))render_editor();
    } else {
      if(tab==="history") {
        const headings=add(list,"div",undefined,"runner-history-headings");
        for(const title of ["Grid","State","","","start","end","T","T_est",""])add(headings,"span",title);                                                   // <<< THOG align Grid wall-clock start/end immediately before elapsed and estimated duration
      }
      for(const grid of [...grids].reverse()){
        const failed=grid.runs.find(run=>run.state==="failed");
        const last=failed?.attempts?.at(-1);
        const failure=last ? ` · exit ${last.exit_code??"?"}${last.failure_excerpt ? ` · ${last.failure_excerpt.trim().split("\n").at(-1)}` : ""}` : "";
        const blocked=grid.runs.find(run=>run.state==="blocked" && run.blocking_reason);
        const reason=blocked ? ` · blocked: ${blocked.blocking_reason}` : "";
        const row=tab==="progress"?add(list,"div",undefined,"runner-active-grid-row"):
          tab==="history"?add(list,"div",undefined,"runner-history-grid-row"):list;
        const locations=[...new Set(grid.runs.map(run=>`${run.host_label} GPU ${run.gpu.ordinal}`))].join(", ");
        const entry=button(row,tab==="progress"?`${grid.grid_tag} · ${grid.label} · ${locations} · ${grid.state}`:
          `${grid.grid_tag} · ${grid.label}`,
        ()=>{chosen=grid.grid_id;manual_grid_selection=true;if(tab==="history")last_history_grid=chosen;render();});
        entry.title=`${grid.grid_tag} · ${grid.label} · ${grid.state}${failure}${reason}`;
        entry.classList.toggle("active",chosen===grid.grid_id);
        if(tab==="history") {
          const outcome = history_outcome(grid);
          row.dataset.outcome = outcome;
          entry.classList.add(`runner-outcome-${outcome}`);
          add(row,"span",grid.state,`runner-status runner-outcome-${outcome}`);
          button(row,"Rename",async()=>{
            const label=await rename_dialog(`Rename ${grid.grid_tag}`,grid.label);
            if(label===null || label.trim()===grid.label)return;
            try{await action("rename_grid",{grid_id:grid.grid_id,label});render();}catch(_){/* Error shown above. */}
          }).classList.add("runner-history-rename");
          const deletion=button(row,"Delete",async()=>{
            if(!confirm(`Delete History for ${grid.grid_tag}? Its generated Runner scripts will also be removed. Training logs and model artifacts remain.`))return;
            try{await action("delete_grid_history",{grid_id:grid.grid_id});if(chosen===grid.grid_id)chosen=null;
              if(last_history_grid===grid.grid_id)last_history_grid=null;render();}catch(_){/* Error shown above. */}
          });
          deletion.classList.add("runner-recipe-delete");deletion.disabled=!["completed","failed","cancelled"].includes(grid.state);
          add(row,"span",wall_time(grid.started_at),"runner-history-wall-time").title = grid.started_at ? new Date(grid.started_at).toLocaleString() : "Start time unavailable";                                                                               // <<< THOG durable Grid first-dispatch wall time
          add(row,"span",wall_time(grid.finished_at),"runner-history-wall-time").title = grid.finished_at ? new Date(grid.finished_at).toLocaleString() : "End time unavailable";                                                                              // <<< THOG durable Grid terminal wall time
          const elapsed=add(row,"span",format_duration(grid_elapsed(grid)),"runner-history-time");
          elapsed.title="Execution wall time from first dispatch to terminal state; excludes initial queue wait";
          const upfront=add(row,"span",estimate_range(grid),"runner-history-estimate");
          upfront.title=`Estimate saved at launch · ${grid.estimated_duration?.confidence||"insufficient"} confidence`;
          const diagnostic=add(row,"span",[failure,reason].filter(Boolean).join(" · ").replace(/^ · /,""),"runner-history-error");
          diagnostic.title=diagnostic.textContent;

        } else entry.classList.add(`runner-status-${grid.state}`);
        if(tab==="progress") {
          button(row,"Rename",async()=>{
            const label=await rename_dialog(`Rename ${grid.grid_tag}`,grid.label);
            if(label===null || label.trim()===grid.label)return;
            try{await action("rename_grid",{grid_id:grid.grid_id,label});render();}catch(_){/* Error shown above. */}
          }).classList.add("runner-rename-grid");
          const cleanup=button(row,grid.flush_requested?"Retry Flush":"Kill and Flush",async()=>{
            if(!confirm(`Kill and Flush ${grid.grid_tag}? This force-kills active training without a graceful checkpoint, cancels queued runs and releases its GPUs when verified. Recipe, History, logs and files remain.`))return;
            chosen=grid.grid_id;
            try{await action("kill_flush",{grid_id:grid.grid_id});}catch(_){/* Error shown above. */}
          });
          cleanup.classList.add("runner-kill-flush");
          cleanup.title="Last resort. Force-stop this Grid; retain its Recipe and History.";
        }
      }
      const finished_selection=tab==="multiview" && manual_grid_selection && chosen && !grids.some(item=>item.grid_id===chosen) &&
        snapshot.grids.some(item=>item.grid_id===chosen);
      const latest_running=[...grids].reverse().find(item=>item.state==="running"||item.runs.some(run=>run.state==="running"));
      const chosen_grid=grids.find(item=>item.grid_id===chosen);
      const grid=finished_selection?null:(manual_grid_selection?chosen_grid:null)||
        (tab==="history"?grids.find(item=>item.grid_id===last_history_grid):null)||
        latest_running||chosen_grid||grids.at(-1);
      if(grid){
        chosen=grid.grid_id;
        if(tab==="history")last_history_grid=chosen;
        if(tab==="multiview"){
          if(multiview.dataset.gridMultiview!==grid.grid_id){
            multiview.replaceChildren();multiview.dataset.gridMultiview=grid.grid_id;
            show_multiview_selector(grids,grid.grid_id);
            add(multiview,"h2",`${grid.grid_tag} · ${grid.label} · Multiview`);
            const frame=add(multiview,"iframe",undefined,"runner-multiview-frame");
            frame.title=`Multiview for ${grid.grid_tag}`;
            frame.src=`/?runner_grid_tag=${encodeURIComponent(grid.grid_tag)}&runner_grid_id=${encodeURIComponent(grid.grid_id)}&runner_grid_host=${encodeURIComponent(grid.grid_owner_host_id || snapshot.local_id || "")}`;
            frame.addEventListener("load",()=>setTimeout(()=>frame.contentDocument?.getElementById("workspace_nav")?.click(),50));
          }else show_multiview_selector(grids,grid.grid_id);
        }else if(tab==="log" && detail.querySelector(".runner-event-log")?.dataset.gridId===grid.grid_id){
          update_event_log();
        }else render_grid(grid);
      }else{detail.replaceChildren();
        const empty_text=finished_selection?"The selected Grid has finished. Select another active Grid or inspect it in History.":
          ["progress","multiview"].includes(tab)?"No Grids are currently active":"No Grids in this view";
        if(tab==="multiview"){
          const empty_key=finished_selection?`finished:${chosen}`:"empty";
          if(multiview.dataset.gridMultiview!==empty_key){multiview.replaceChildren();multiview.dataset.gridMultiview=empty_key;}
          if(grids.length)show_multiview_selector(grids,null);
          const note=multiview.querySelector(".runner-empty-state");
          if(note)note.textContent=empty_text;
          else add(multiview,"p",empty_text,"runner-empty-state");
        }else add(detail,"p",empty_text,"runner-empty-state");}
    }
    view.classList.toggle("runner-full-width",tab==="multiview");
    view.classList.toggle("runner-history",tab==="history");
    multiview.hidden=tab!=="multiview";
  }
  async function refresh(repaint=false) {
    if(!visible || polling)return;
    polling=true;
    try {
      const [next,hosts]=await Promise.all([request("/api/runner"),request("/api/network")]);
      snapshot=next;network=hosts;
      const newest=next.grids.at(-1)?.grid_id||null;
      if(newest && newest!==last_seen_grid && !last_history_grid)last_history_grid=newest;
      last_seen_grid=newest;
      if(repaint||tab!=="recipes"||!dirty)render();
      if(!metrics_loading) {
        metrics_loading=true;
        request("/api/runs").then(run_catalogue=>{
          current_run_metrics=new Map((run_catalogue.runs||[]).filter(item=>item.runner_run_id)
            .map(item=>[item.runner_run_id,item]));
          if(visible && (tab==="history" || tab==="progress"))render();
        }).catch(()=>{}).finally(()=>{metrics_loading=false;});
      }
    } catch(error){message.textContent=error.message;clearTimeout(message_timer);message_timer=setTimeout(clear_message,18000);}
    finally{polling=false;}
  }
  by_id("runner_nav").addEventListener("click",()=>{
    if(!visible){manual_grid_selection=false;chosen=null;}
    visible=true;refresh(true);
  });
  for(const id of ["runs_nav","workspace_nav","networks_nav","settings_nav"])by_id(id)?.addEventListener("click",()=>{visible=false;});
  for(const control of by_id("runner_tabs").querySelectorAll("button"))control.addEventListener("click",()=>{
    clear_message();required_notice_seen=true;
    const was_active=["progress","multiview"].includes(tab);
    tab=control.dataset.runnerTab;
    if(!was_active)manual_grid_selection=false;
    chosen=tab==="history"?last_history_grid:["progress","multiview"].includes(tab) && was_active?chosen:null;
    render();
  });
  setInterval(()=>{if(visible && tab!=="recipes")refresh();},5000);
  window.instra_runner_test_hooks = Object.freeze({recipe_problems,apply_width_selection,current_recipe,remember_default,format_duration,grid_elapsed,estimate_range,estimated_run_end,history_outcome,field_help,invalid_field_value,premat_enabled,category_enabled,categories_for_field,matches_search,compare_fields});
})();
// ^^^ THOG
