// vvv THOG render the four Runner views from validated backend state without submitting shell text
"use strict";
(function install_runner() {
  const by_id = name => document.getElementById(name);
  const view = by_id("runner_view");
  if (!view) return;
  const list = by_id("runner_list"), detail = by_id("runner_detail"), message = by_id("runner_message");
  const categories = ["Run Control Parameters", "Base Hyperparameters", "Geometry", "Premat", "NSIGHT", "Plastic", "Chaos Bumps"];
  let snapshot = null, network = null, tab = "recipes", chosen = null, category = "Common";
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
  function edit(key, value) { draft = current_recipe(); draft.parameters[key] = value; dirty = true; }
  function parse_value(key, raw, spec) {
    if (spec.kind === "list") return raw.split("\n").filter(Boolean);
    const scalar = value => spec.type === "int" || spec.type === "float" ? Number(value) :
      spec.type === "flag" && ["true","false"].includes(value) ? value === "true" : value;
    return spec.kind === "dimension" ? raw.split(",").map(value => scalar(value.trim())) : scalar(raw);
  }
  function show_fields(container, keys) {
    const grid = add(container,"div",undefined,"runner-fields");
    for (const key of keys) {
      const spec = snapshot.catalogue[key];
      if (!spec || ["manual","automatic"].includes(spec.kind)) continue;
      const label = add(grid,"label",key);
      const field = add(label,"input");
      field.value = Array.isArray(current_recipe().parameters[key]) ? current_recipe().parameters[key].join(spec.kind === "list" ? "\n" : ", ") :
        String(current_recipe().parameters[key] ?? "");
      field.placeholder = spec.kind === "dimension" ? "One or comma-separated choices" : spec.kind === "list" ? "One item per line" :
        spec.type === "flag" ? "true or false" : String(spec.default ?? "");
      field.title = `${spec.category} · ${spec.kind} · ${spec.type}`;
      field.addEventListener("change", () => {
        if (field.value === "") { delete current_recipe().parameters[key]; dirty = true; }
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
    label_input.addEventListener("change", () => { draft = current_recipe(); draft.label = label_input.value; dirty = true; });
    const controls = add(detail,"div",undefined,"runner-actions");
    button(controls,"Save",async () => {
      try { const saved = await action("save",{recipe_id:draft_id,recipe:current_recipe()}); draft_id=saved.recipe_id; dirty=false; chosen=draft_id; render(); }
      catch (_) { /* The error is displayed above. */ }
    });
    button(controls,"Preview",async () => {
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
    parallel.addEventListener("change",()=>{draft=current_recipe();draft.max_parallel=Number(parallel.value);dirty=true;});
    const profile_label=add(selectors,"label","Profiling");
    const profile=add(profile_label,"select");
    for (const [name,value] of [["None","none"],["NSYS","nsys"],["NCU","ncu"],["NSYS and NCU pair","pair"]]) {
      const option=add(profile,"option",name);option.value=value;
    }
    profile.value=recipe.profilers?.length===2?"pair":recipe.profilers?.[0]||"none";
    profile.addEventListener("change",()=>{draft=current_recipe();draft.profilers=profile.value==="pair"?["nsys","ncu"]:[profile.value];dirty=true;});
    const pool = add(detail,"details"); add(pool,"summary","Eligible hosts and GPUs");
    for (const host of network?.hosts || []) {
      if (!host.local && !host.execution_enabled) continue;
      for (const gpu of host.last_discovered?.gpus || []) {
        const gpu_id=gpu.gpu_id||`${host.thog_host_id}.gpu.${gpu.gpu_key}`;
        const line=add(pool,"label",`${host.display_name} GPU ${gpu.ordinal} · ${gpu.model} · ${gpu.memory_mib} MiB`);
        const tick=add(line,"input");tick.type="checkbox";
        tick.checked=(recipe.gpu_pool||[]).includes(gpu_id);
        tick.addEventListener("change",()=>{draft=current_recipe();draft.gpu_pool=draft.gpu_pool||[];
          draft.gpu_pool=tick.checked?[...draft.gpu_pool,gpu_id]:draft.gpu_pool.filter(id=>id!==gpu_id);dirty=true;});
        const watts=add(pool,"label",`Power cap for ${host.display_name} GPU ${gpu.ordinal} (W; blank keeps its current policy)`);
        const cap=add(watts,"input");cap.type="number";cap.min="50";cap.max="600";cap.value=recipe.power_caps?.[gpu_id]??"";
        cap.addEventListener("change",()=>{draft=current_recipe();draft.power_caps=draft.power_caps||{};
          if(cap.value)draft.power_caps[gpu_id]=Number(cap.value);else delete draft.power_caps[gpu_id];dirty=true;});
      }
    }
    add(pool,"p","If no GPU is selected, Runner uses eligible GPUs on permitted hosts. One-run Grids use one GPU by default.");
    const search=add(detail,"input");search.placeholder="Search every parameter category";
    const categories_row=add(detail,"nav",undefined,"runner-categories");
    const fields=add(detail,"div");
    function show_category(query="") {
      fields.replaceChildren();categories_row.replaceChildren();
      for (const name of ["Common",...categories]) button(categories_row,name,()=>{category=name;show_category(search.value);})
        .classList.toggle("active",category===name);
      const keys=(query?Object.keys(snapshot.catalogue).filter(key => key.toLowerCase().includes(query.toLowerCase())):
        category==="Common"?snapshot.common:Object.keys(snapshot.catalogue).filter(key=>snapshot.catalogue[key].category===category));
      show_fields(fields,keys);
    }
    search.addEventListener("input",()=>show_category(search.value));show_category();
    add(detail,"section",undefined).id="runner_preview";
  }
  function render_run(parent,run) {
    const row=add(parent,"details",undefined,"runner-run");
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
    for (const attempt of run.attempts||[]) add(row,"p",`Attempt ${attempt.attempt_id} · ${attempt.started_at} · ${attempt.state||run.state}`);
    add(row,"pre",JSON.stringify(run.parameters,null,2));
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
    if (tab==="files") {
      for (const name of ["script","manifest","placement","status",...(grid.conversion?["conversion"]:[])]) {
        const path=`/api/runner/file?grid_id=${encodeURIComponent(grid.grid_id)}&name=${name}`;
        const link=add(detail,"a",`Download ${name}`);link.href=path+"&download=1";link.download="";
        detail.append(document.createElement("br"));
      }
      const viewer=add(detail,"pre");
      fetch(`/api/runner/file?grid_id=${encodeURIComponent(grid.grid_id)}&name=script`).then(response=>response.text())
        .then(text=>{if(chosen===grid.grid_id && tab==="files")viewer.textContent=text;});
    }
    for (const run of grid.runs) if(tab!=="files")render_run(detail,run);
  }
  function render() {
    if (!snapshot || !visible) return;
    for (const control of by_id("runner_tabs").querySelectorAll("button"))control.classList.toggle("active",control.dataset.runnerTab===tab);
    list.replaceChildren();
    const grids=snapshot.grids.filter(grid=>tab!=="progress"||!["completed","failed","cancelled"].includes(grid.state));
    if(tab==="recipes") {
      button(list,"Add Grid Recipe",()=>{draft_id=null;draft=null;chosen=null;dirty=false;render_editor();});
      for(const saved of snapshot.recipes)button(list,saved.recipe.label,()=>{draft_id=saved.recipe_id;chosen=draft_id;
        draft=JSON.parse(JSON.stringify(saved.recipe));dirty=false;render();}).classList.toggle("active",chosen===saved.recipe_id);
      if(chosen && !snapshot.recipes.some(item=>item.recipe_id===chosen))chosen=null;
      if(!dirty || !detail.querySelector("input:focus"))render_editor();
    } else {
      for(const grid of [...grids].reverse())button(list,`${grid.grid_tag} · ${grid.label} · ${grid.state}`,
        ()=>{chosen=grid.grid_id;render();}).classList.toggle("active",chosen===grid.grid_id);
      const grid=grids.find(item=>item.grid_id===chosen)||grids.at(-1);
      if(grid){chosen=grid.grid_id;render_grid(grid);}else{detail.replaceChildren();add(detail,"p","No Grids in this view");}
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
})();
// ^^^ THOG
