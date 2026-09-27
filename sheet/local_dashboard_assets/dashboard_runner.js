// vvv THOG render the four Runner views from validated backend state without submitting shell text
"use strict";
(function install_runner() {
  const by_id = name => document.getElementById(name);
  const view = by_id("runner_view");
  if (!view) return;
  const list = by_id("runner_list"), detail = by_id("runner_detail"), message = by_id("runner_message");
  const categories = ["Run Control Parameters", "GPT-2 Hyperparameters", "Geometry", "Premat", "NSIGHT",
    "Coarse", "Layer Spacing", "Variable Depth", "Chaos Bumps"];
  const main_table_order = ["--geometry-preset", "--optimizer", "--n-layer", "DEPTH.order", "--warmup-iters",
    "--block-size", "--n-embd", "--n-head", "--gradient-accumulation-steps", "--checkpoint-segment-size",
    "--learning-rate", "--min-lr", "--max-iters", "--batch-size"];
  const required_parameters = ["--geometry-preset", "--optimizer", "--n-layer", "--warmup-iters", "--block-size",
    "--n-embd", "--n-head", "--gradient-accumulation-steps", "--checkpoint-segment-size",
    "--learning-rate", "--min-lr", "--max-iters", "--batch-size"];
  let snapshot = null, network = null, tab = "recipes", chosen = null, category = "Frequently Used";
  let draft = null, draft_id = null, dirty = false, visible = false, polling = false;
  const add = (parent, tag, value, class_name) => {
    const element = document.createElement(tag);
    if (value !== undefined) element.textContent = String(value ?? "—");
    if (class_name) element.className = class_name;
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
    } finally { clearTimeout(timeout); }
  }
  async function action(name, values) {
    message.textContent = `${name}…`;
    try {
      const result = await request("/api/runner/action", {method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({action:name, ...values})});
      message.textContent = `${name} complete`; await refresh(true); return result;
    } catch (error) { message.textContent = error.message; throw error; }
  }
  function current_recipe() {
    return draft || {label:"New Grid Recipe", parameters:{"--max-iters":50,"--batch-size":16,"--geometry-preset":"depth",
      "--n-layer":16,"DEPTH.order":12,"--n-embd":1024,"--n-head":16,"--block-size":1024,
      "--gradient-accumulation-steps":6,"--checkpoint-segment-size":4,"--optimizer":"adamw",
      "--learning-rate":.0009,"--min-lr":.00009,"--warmup-iters":0}, max_parallel:1, profilers:["none"]};
  }
  function recipe_problems(recipe, hosts) {
    const errors = [];
    if (!String(recipe.label || "").trim()) errors.push("Recipe label is required");
    const parameters = recipe.parameters || {};
    for (const key of required_parameters) {
      if (parameters[key] === undefined || parameters[key] === null || parameters[key] === "" ||
          Array.isArray(parameters[key]) && !parameters[key].length) errors.push(`${key} is required`);
    }
    if (parameters["--geometry-preset"] !== "dense" && parameters["--model-type"] !== "dense" &&
        (parameters["DEPTH.order"] === undefined || parameters["DEPTH.order"] === "")) {
      errors.push("DEPTH.order is required for a DEPTH Recipe");
    }
    if (!Number.isInteger(Number(recipe.max_parallel)) || Number(recipe.max_parallel) < 1 || Number(recipe.max_parallel) > 64)
      errors.push("Maximum simultaneous runs must be between 1 and 64");
    if (!Array.isArray(recipe.profilers) || !recipe.profilers.length) errors.push("Profiling mode is required");
    const eligible = (hosts || []).filter(host => host.local || host.execution_enabled)
      .flatMap(host => (host.last_discovered?.execution_profiles?.length ? host.last_discovered.gpus || [] : [])
        .map(gpu => gpu.gpu_id || `${host.thog_host_id}.gpu.${gpu.gpu_key}`));
    if (!eligible.length) errors.push("No discovered host with an execution profile and GPU is eligible; check Networks");
    for (const gpu_id of recipe.gpu_pool || []) {
      if (!eligible.includes(gpu_id)) errors.push(`Selected GPU ${gpu_id} is no longer eligible`);
    }
    for (const key of required_parameters) {
      const values = Array.isArray(parameters[key]) ? parameters[key] : [parameters[key]];
      if (values.some(value => value !== undefined && value !== "" &&
          ["--geometry-preset", "--optimizer"].includes(key) === false && !Number.isFinite(Number(value))))
        errors.push(`${key} must contain valid numbers`);
    }
    return errors;
  }
  function check_recipe() {
    const errors = recipe_problems(current_recipe(), network?.hosts);
    const notice = by_id("runner_required_fields");
    if (notice) {
      notice.replaceChildren();
      add(notice,"strong",errors.length ? `Complete ${errors.length} required item${errors.length === 1 ? "" : "s"} before saving, previewing or launching:` :
        "Required inputs complete. Placement is checked again during Preview.");
      if (errors.length) {
        const items = add(notice,"ul");
        for (const error of errors) add(items,"li",error);
      }
    }
    if (errors.length) message.textContent = errors.join("; ");
    return !errors.length;
  }
  function edit(key, value) { draft = current_recipe(); draft.parameters[key] = value; dirty = true; check_recipe(); }
  function parse_value(key, raw, spec) {
    if (spec.kind === "list") return raw.split("\n").filter(Boolean);
    const scalar = value => spec.type === "int" || spec.type === "float" ? Number(value) :
      spec.type === "flag" && ["true","false"].includes(value) ? value === "true" : value;
    return spec.kind === "dimension" ? raw.split(",").map(value => scalar(value.trim())) : scalar(raw);
  }
  function show_fields(container, keys, include_profiler=false) {
    const grid = add(container,"div",undefined,"runner-fields");
    const columns = window.innerWidth < 800 ? 1 : window.innerWidth < 1300 ? 2 : category === "Frequently Used" && !by_id("runner_parameter_search")?.value ? 5 : 3;
    grid.style.setProperty("--runner-columns",String(columns));
    grid.style.setProperty("--runner-rows",String(Math.max(1,Math.ceil((keys.length + (include_profiler ? 1 : 0))/columns))));
    if (include_profiler) {
      const profile_label=add(grid,"label","Profiling mode");
      profile_label.title=snapshot.catalogue["--premat_processing_profiler"]?.help || "Select NSYS, NCU or a paired capture";
      const profile=add(profile_label,"select");
      for (const [name,value] of [["None","none"],["NSYS","nsys"],["NCU","ncu"],["NSYS and NCU pair","pair"]]) {
        const option=add(profile,"option",name);option.value=value;
      }
      profile.value=current_recipe().profilers?.length===2?"pair":current_recipe().profilers?.[0]||"none";
      profile.addEventListener("change",()=>{draft=current_recipe();draft.profilers=profile.value==="pair"?["nsys","ncu"]:[profile.value];dirty=true;});
    }
    for (const key of keys) {
      const spec = snapshot.catalogue[key];
      if (!spec || spec.ui_hidden || ["manual","automatic"].includes(spec.kind)) continue;
      const label = add(grid,"label",`${key}${spec.short ? ` (${spec.short})` : ""}`);
      label.title = spec.help;
      const field = add(label,"input");
      field.title=spec.help;
      field.value = Array.isArray(current_recipe().parameters[key]) ? current_recipe().parameters[key].join(spec.kind === "list" ? "\n" : ", ") :
        String(current_recipe().parameters[key] ?? "");
      field.placeholder = spec.kind === "dimension" ? "One or comma-separated choices" : spec.kind === "list" ? "One item per line" :
        spec.type === "flag" ? "true or false" : String(spec.default ?? "");
      field.addEventListener("change", () => {
        if (field.value === "") { draft=current_recipe();delete draft.parameters[key];dirty=true;check_recipe(); }
        else edit(key, parse_value(key, field.value, spec));
      });
    }
  }
  function render_editor() {
    detail.replaceChildren();
    const recipe = current_recipe();
    add(detail,"h2", draft_id ? `Edit Recipe · ${recipe.label}` : "Add Grid Recipe");
    const label = add(detail,"label","Recipe label");
    const label_input = add(label,"input"); label_input.value = recipe.label;
    label_input.addEventListener("change", () => { draft = current_recipe(); draft.label = label_input.value; dirty = true;check_recipe(); });
    add(detail,"div",undefined,"runner-required-fields").id="runner_required_fields";
    const controls = add(detail,"div",undefined,"runner-actions");
    button(controls,"Save",async () => {
      if (!check_recipe()) return;
      try { const saved = await action("save",{recipe_id:draft_id,recipe:current_recipe()}); draft_id=saved.recipe_id; dirty=false; chosen=draft_id; render(); }
      catch (_) { /* The error is displayed above. */ }
    });
    button(controls,"Preview",async () => {
      if (!check_recipe()) return;
      try {
        const preview = await action("preview",{recipe:current_recipe()});
        const panel = by_id("runner_preview");
        panel.replaceChildren();
        add(panel,"h3",`Total Runs in Grid: ${preview.total_runs} · physical executions: ${preview.runs.length}`);
        add(panel,"p", preview.estimated_duration.seconds === null ? preview.estimated_duration.explanation :
          `Estimate ${preview.estimated_duration.seconds}s (${preview.estimated_duration.interval_seconds.join("–")}s), ${preview.estimated_duration.confidence}, ${preview.estimated_duration.exemplars} exemplars`);
        for(const place of preview.gpu_pool)add(panel,"p",`${place.host_label} GPU ${place.gpu.ordinal} · free ${place.gpu.free_mib??"unknown"} MiB · reservation ${place.reservation_owner?.grid_id||"none"}`);
        for (const run of preview.runs) render_run(panel,run);
      } catch (_) { /* The error is displayed above. */ }
    });
    button(controls,"Launch",async () => {
      if (!check_recipe()) return;
      if (!draft_id || dirty) { message.textContent="Save the Recipe before launching its snapshot"; return; }
      try {
        const preview = await action("preview",{recipe:current_recipe()});
        const names = preview.gpu_pool.map(item => `${item.host_label} GPU ${item.gpu.ordinal}`).join(", ");
        if (!confirm(`Launch ${preview.total_runs} THOG runs on ${names}?`)) return;
        await action("launch",{recipe_id:draft_id,confirm_large:true}); tab="progress"; render();
      } catch (_) { /* The error is displayed above. */ }
    });
    const selectors = add(detail,"div",undefined,"runner-fields");
    const max_label=add(selectors,"label","Maximum simultaneous runs");
    const parallel=add(max_label,"input"); parallel.type="number"; parallel.min="1"; parallel.max="64";
    parallel.value=recipe.max_parallel||1;
    parallel.addEventListener("change",()=>{draft=current_recipe();draft.max_parallel=Number(parallel.value);dirty=true;check_recipe();});
    max_label.title="Maximum number of Grid runs that may execute at once";
    parallel.title=max_label.title;
    const eligible=(network?.hosts||[]).filter(host=>(host.local||host.execution_enabled) &&
      host.last_discovered?.execution_profiles?.length)
      .flatMap(host=>(host.last_discovered?.gpus||[]).map(gpu=>({host,gpu})));
    const placement=add(detail,"p",`GPU placement: ${recipe.gpu_pool?.length ? `${recipe.gpu_pool.length} selected GPU${recipe.gpu_pool.length===1?"":"s"}` : `Automatic (default) · ${eligible.length} discovered GPU${eligible.length===1?"":"s"}`}` ,"runner-placement-status");
    const pool = add(detail,"details",undefined,"runner-gpu-pool");
    const placement_text = selected => selected.length ? `${selected.length} selected` : `Automatic (default) from ${eligible.length} discovered GPU${eligible.length===1?"":"s"}`;
    add(pool,"summary",`Eligible hosts and GPUs · ${placement_text(recipe.gpu_pool||[])}`);
    add(pool,"p","Select GPUs to restrict placement. With none selected, Runner chooses from eligible hosts.","runner-gpu-note");
    for (const host of network?.hosts || []) {
      if ((!host.local && !host.execution_enabled) || !host.last_discovered?.execution_profiles?.length) continue;
      for (const gpu of host.last_discovered?.gpus || []) {
        const gpu_id=gpu.gpu_id||`${host.thog_host_id}.gpu.${gpu.gpu_key}`;
        const line=add(pool,"div",undefined,"runner-gpu-row");
        const tick=add(line,"input");tick.type="checkbox";tick.setAttribute("aria-label",`Select ${host.display_name} GPU ${gpu.ordinal}`);
        tick.checked=(recipe.gpu_pool||[]).includes(gpu_id);
        tick.addEventListener("change",()=>{draft=current_recipe();draft.gpu_pool=draft.gpu_pool||[];
          draft.gpu_pool=tick.checked?[...draft.gpu_pool,gpu_id]:draft.gpu_pool.filter(id=>id!==gpu_id);dirty=true;
          pool.querySelector("summary").textContent=`Eligible hosts and GPUs · ${placement_text(draft.gpu_pool)}`;
          placement.textContent=`GPU placement: ${draft.gpu_pool.length ? placement_text(draft.gpu_pool) : placement_text([])}`;check_recipe();});
        add(line,"span",`${host.display_name} · GPU ${gpu.ordinal} · ${gpu.model} · ${gpu.memory_mib??"?"} MiB`);
        const watts=add(line,"label","Power cap (W)");
        watts.title="Blank keeps this GPU's current power policy";
        const cap=add(watts,"input");cap.type="number";cap.min="50";cap.max="600";cap.value=recipe.power_caps?.[gpu_id]??"";
        cap.title=watts.title;
        cap.addEventListener("change",()=>{draft=current_recipe();draft.power_caps=draft.power_caps||{};
          if(cap.value)draft.power_caps[gpu_id]=Number(cap.value);else delete draft.power_caps[gpu_id];dirty=true;});
      }
    }
    const search=add(detail,"input",undefined,"runner-parameter-search");search.id="runner_parameter_search";
    search.placeholder="Search fields across all parameter tabs";
    const categories_row=add(detail,"nav",undefined,"runner-categories");
    const fields=add(detail,"div");
    function show_category(query="") {
      fields.replaceChildren();categories_row.replaceChildren();
      for (const name of ["Frequently Used",...categories]) button(categories_row,name,()=>{category=name;show_category(search.value);})
        .classList.toggle("active",category===name);
      const keys=(query?Object.keys(snapshot.catalogue).filter(key => key.toLowerCase().includes(query.toLowerCase()) ||
        snapshot.catalogue[key].help?.toLowerCase().includes(query.toLowerCase())):
        category==="Frequently Used"?snapshot.common:Object.keys(snapshot.catalogue).filter(key=>snapshot.catalogue[key].category===category))
        .filter(key=>!snapshot.catalogue[key].ui_hidden && !["manual","automatic"].includes(snapshot.catalogue[key].kind))
        .sort((a,b)=>{const left=main_table_order.indexOf(a),right=main_table_order.indexOf(b);
          return (left<0?999:left)-(right<0?999:right) || a.localeCompare(b);});
      show_fields(fields,keys,category==="NSIGHT" && !query);
    }
    search.addEventListener("input",()=>show_category(search.value));show_category();
    add(detail,"section",undefined).id="runner_preview";
    check_recipe();
  }
  function render_run(parent,run) {
    const row=add(parent,"details",undefined,"runner-run");
    if (run.state === "failed") row.open = true;
    add(row,"summary",`${run.run_id.slice(0,8)} · ${run.state} · ${run.host_label} GPU ${run.gpu.ordinal} · ${run.profiler.toUpperCase()}`);
    add(row,"p",`GPU ${run.gpu.model} · ${run.gpu.uuid||run.gpu.gpu_key} · ${run.execution_profile} · ${run.dtype}/${run.attention_backend} · peak ${run.required_mib} MiB · power requested ${run.requested_power_w??"default"} W`);
    if (run.pairing_id) add(row,"p",`Paired profiler runs: ${run.pairing_id}`);
    if (run.blocking_reason) add(row,"p",run.blocking_reason);
    if (run.attempts?.length) {
      const started=Date.parse(run.attempts.at(-1).started_at);
      const seconds=run.duration_seconds??Math.max(0,Math.round((Date.now()-started)/1000));
      add(row,"p",`Elapsed ${seconds}s · attempts ${run.attempts.length}`);
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
      if (attempt.failure_excerpt) add(row,"pre",attempt.failure_excerpt,"runner-failure-log");
      if (attempt.state==="failed" || run.state==="failed") {
        const log=add(row,"pre","","runner-failure-log");log.hidden=true;
        button(row,"View attempt log",async()=>{
          try {
            const query=new URLSearchParams({grid_id:run.grid_id,run_id:run.run_id,attempt_id:attempt.attempt_id});
            const result=await request(`/api/runner/log?${query}`);
            log.textContent=result.text||"No log output";log.hidden=false;
          } catch(error){message.textContent=error.message;}
        });
      }
    }
    add(row,"pre",JSON.stringify(run.parameters,null,2));
  }
  function render_script(parent,grid) {
    const script_url=`/api/runner/file?grid_id=${encodeURIComponent(grid.grid_id)}&name=script`;
    const link=add(parent,"a",`Download ${grid.grid_tag} Bash script`);
    link.href=script_url+"&download=1";link.download="";
    const manifest=add(parent,"a","View resolved manifest");
    manifest.href=`/api/runner/file?grid_id=${encodeURIComponent(grid.grid_id)}&name=manifest`;
    manifest.target="_blank";manifest.rel="noopener";
    add(parent,"p","The Bash export calls the existing Python training entry point for every resolved run. There is no separate generated Python script.");
    const viewer=add(parent,"pre","Loading script…","runner-script-viewer");
    fetch(script_url).then(async response=>{if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.text();})
      .then(source=>{if(chosen===grid.grid_id && ["files","current_scripts"].includes(tab))viewer.textContent=source;})
      .catch(error=>{viewer.textContent=`Script unavailable: ${error.message}`;});
  }
  function render_grid(grid) {
    detail.replaceChildren();add(detail,"h2",`${grid.grid_tag} · ${grid.label}`);
    add(detail,"p",`State ${grid.state} · ${grid.runs.length} runs · Recipe ${grid.recipe_id} · started ${new Date(grid.created_at).toLocaleString()}`);
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
          const estimate=proposed.estimated_duration.seconds===null?"insufficient history":`${proposed.estimated_duration.seconds}s`;
          if(!confirm(`Repeat ${grid.grid_tag} (${mode})?\nRecipe: ${proposed.recipe.label}\nRuns: ${proposed.total_runs}; executions: ${proposed.physical_executions}\nPlacement: ${places}\nEstimate: ${estimate}\nChanges: ${differences||"none"}${proposed.changes.length>8?"\nMore changes appear in the resolved preview":""}`))return;
          await action("repeat",{grid_id:grid.grid_id,mode,confirm_large:true});tab="progress";render();
        }
        catch (_) { /* Error shown above. */ }
      });
    }
    if (tab==="files" || tab==="current_scripts") {
      if (tab==="files") for (const name of ["manifest","placement","status",...(grid.conversion?["conversion"]:[])]) {
        const path=`/api/runner/file?grid_id=${encodeURIComponent(grid.grid_id)}&name=${name}`;
        const link=add(detail,"a",`Download ${name}`);link.href=path+"&download=1";link.download="";
        detail.append(document.createElement("br"));
      }
      render_script(detail,grid);
    }
    for (const run of grid.runs) if(!["files","current_scripts"].includes(tab))render_run(detail,run);
  }
  function render() {
    if (!snapshot || !visible) return;
    for (const control of by_id("runner_tabs").querySelectorAll("button"))control.classList.toggle("active",control.dataset.runnerTab===tab);
    list.replaceChildren();
    const grids=snapshot.grids.filter(grid=>!["progress","current_scripts"].includes(tab)||!["completed","failed","cancelled"].includes(grid.state));
    if(tab==="recipes") {
      button(list,"Add Grid Recipe",()=>{
        if(dirty && !confirm("Discard unsaved Recipe edits and start a new Recipe?"))return;
        draft_id=null;draft=null;chosen=null;dirty=false;render_editor();
      }).classList.add("runner-add-recipe");
      for(const saved of snapshot.recipes)button(list,saved.recipe.label,()=>{draft_id=saved.recipe_id;chosen=draft_id;
        draft=JSON.parse(JSON.stringify(saved.recipe));dirty=false;render();}).classList.toggle("active",chosen===saved.recipe_id);
      if(chosen && !snapshot.recipes.some(item=>item.recipe_id===chosen))chosen=null;
      if(!dirty || !detail.querySelector("input:focus"))render_editor();
    } else {
      for(const grid of [...grids].reverse()){
        const failed=grid.runs.find(run=>run.state==="failed");
        const last=failed?.attempts?.at(-1);
        const failure=last ? ` · exit ${last.exit_code??"?"}${last.failure_excerpt ? ` · ${last.failure_excerpt.trim().split("\n").at(-1).slice(0,110)}` : ""}` : "";
        button(list,`${grid.grid_tag} · ${grid.label} · ${grid.state}${failure}`,
        ()=>{chosen=grid.grid_id;render();}).classList.toggle("active",chosen===grid.grid_id);
      }
      const grid=grids.find(item=>item.grid_id===chosen)||grids.at(-1);
      if(grid){chosen=grid.grid_id;render_grid(grid);}else{detail.replaceChildren();add(detail,"p",
        ["progress","current_scripts"].includes(tab)?"No Grids are currently active":"No Grids in this view");}
    }
  }
  async function refresh(repaint=false) {
    if(!visible || polling)return;
    polling=true;
    try {
      const [next,hosts]=await Promise.all([request("/api/runner"),request("/api/network")]);
      snapshot=next;network=hosts;
      if(repaint||tab!=="recipes"||!dirty)render();
    } catch(error){message.textContent=error.message;}
    finally{polling=false;}
  }
  by_id("runner_nav").addEventListener("click",()=>{visible=true;refresh(true);});
  for(const id of ["runs_nav","workspace_nav","networks_nav","settings_nav"])by_id(id)?.addEventListener("click",()=>{visible=false;});
  for(const control of by_id("runner_tabs").querySelectorAll("button"))control.addEventListener("click",()=>{
    tab=control.dataset.runnerTab;chosen=null;render();
  });
  setInterval(()=>{if(visible && tab!=="recipes")refresh();},5000);
  window.instra_runner_test_hooks = Object.freeze({recipe_problems});
})();
// ^^^ THOG
