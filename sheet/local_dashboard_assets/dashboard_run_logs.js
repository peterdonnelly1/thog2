// vvv THOG bounded incremental ANSI console viewer; requests cancel on run and view changes
"use strict";
window.addEventListener("load",()=>{
  setTimeout(()=>{
    const ansi_basic = [
      "#000000", "#cd0000", "#00cd00", "#cdcd00",
      "#0000ee", "#cd00cd", "#00cdcd", "#e5e5e5",
      "#7f7f7f", "#ff0000", "#00ff00", "#ffff00",
      "#5c5cff", "#ff00ff", "#00ffff", "#ffffff",
    ];

    const ansi_256_colour = value => {
      const index = Math.max(0, Math.min(255, Number(value) || 0));
      if (index < 16) return ansi_basic[index];
      if (index >= 232) {
        const shade = 8 + (index - 232) * 10;
        return `rgb(${shade},${shade},${shade})`;
      }
      const cube = index - 16;
      const blue = cube % 6;
      const green = Math.floor(cube / 6) % 6;
      const red = Math.floor(cube / 36) % 6;
      const component = level => level === 0 ? 0 : 55 + level * 40;
      return `rgb(${component(red)},${component(green)},${component(blue)})`;
    };

    const clean_terminal_text = value => String(value)
      .replace(/\x1b\][^\x07]*(?:\x07|\x1b\\)/g, "")
      .replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");

    const apply_sgr = (parameters, state) => {
      const values = parameters.length ? parameters.map(value => Number(value || 0)) : [0];
      for (let index = 0; index < values.length; index += 1) {
        const code = values[index];
        if (code === 0) {
          state.colour = null; state.bold = false; state.dim = false; state.italic = false; state.underline = false;
        } else if (code === 1) state.bold = true;
        else if (code === 2) state.dim = true;
        else if (code === 3) state.italic = true;
        else if (code === 4) state.underline = true;
        else if (code === 22) { state.bold = false; state.dim = false; }
        else if (code === 23) state.italic = false;
        else if (code === 24) state.underline = false;
        else if (code === 39) state.colour = null;
        else if (code >= 30 && code <= 37) state.colour = ansi_basic[code - 30];
        else if (code >= 90 && code <= 97) state.colour = ansi_basic[8 + code - 90];
        else if (code === 38 && values[index + 1] === 5 && Number.isFinite(values[index + 2])) {
          state.colour = ansi_256_colour(values[index + 2]); index += 2;
        } else if (code === 38 && values[index + 1] === 2 && values.slice(index + 2, index + 5).every(Number.isFinite)) {
          const [red, green, blue] = values.slice(index + 2, index + 5).map(value => Math.max(0, Math.min(255, value)));
          state.colour = `rgb(${red},${green},${blue})`; index += 4;
        }
      }
    };

    const ansi_line = value => {
      const line = document.createElement("div");
      line.className = "local-log-line";
      const state = {colour: null, bold: false, dim: false, italic: false, underline: false};
      const expression = /\x1b\[([0-9;]*)m/g;
      let cursor = 0;
      let match = null;
      const append = text => {
        const cleaned = clean_terminal_text(text);
        if (!cleaned) return;
        const span = document.createElement("span");
        span.textContent = cleaned;
        if (state.colour) span.style.color = state.colour;
        if (state.bold) span.style.fontWeight = "700";
        if (state.dim) span.style.opacity = "0.72";
        if (state.italic) span.style.fontStyle = "italic";
        if (state.underline) span.style.textDecoration = "underline";
        line.appendChild(span);
      };
      while ((match = expression.exec(value)) !== null) {
        append(value.slice(cursor, match.index));
        apply_sgr(match[1].split(";"), state);
        cursor = expression.lastIndex;
      }
      append(value.slice(cursor));
      if (!line.childNodes.length) line.appendChild(document.createTextNode(" "));
      return line;
    };

    const log_state={run_id:null,path:"",offset:null,partial:"",partial_node:null,controller:null,user_scrolled:false};
    let log_font_size=Number(localStorage.getItem("thog2_local_log_font_size")) || 12;
    const pane=document.createElement("section");pane.id="run_logs_pane";pane.hidden=true;pane.className="run-logs-pane";
    pane.innerHTML='<header class="local-log-toolbar"><span id="local_log_status">Select Logs to load console output.</span><div><button type="button" id="local_log_smaller">A↓</button><span id="local_log_font_value"></span><button type="button" id="local_log_larger">A↑</button></div></header><div id="local_log_output" role="log" aria-live="off"></div>';
    by_id("charts_pane").appendChild(pane);
    const output=by_id("local_log_output"),status=by_id("local_log_status");
    function font(value){log_font_size=Math.max(8,Math.min(24,value));localStorage.setItem("thog2_local_log_font_size",String(log_font_size));output.style.fontSize=`${log_font_size}px`;by_id("local_log_font_value").textContent=` ${log_font_size}px `;}
    font(log_font_size);by_id("local_log_smaller").onclick=()=>font(log_font_size-1);by_id("local_log_larger").onclick=()=>font(log_font_size+1);
    output.addEventListener("scroll",()=>{log_state.user_scrolled=output.scrollTop>12;});
    function reset_log(){
      log_state.controller?.abort();log_state.controller=null;log_state.run_id=app.current_run_id;
      log_state.path="";log_state.offset=null;log_state.partial="";log_state.partial_node=null;log_state.user_scrolled=false;
      output.replaceChildren();output.scrollTop=0;status.textContent="Loading console output…";
    }
    function ingest_log(text,reset){
      if(reset){output.replaceChildren();log_state.partial="";log_state.partial_node=null;log_state.user_scrolled=false;}
      log_state.partial_node?.remove();log_state.partial_node=null;
      const lines=(log_state.partial+String(text || "")).replace(/\r\n/g,"\n").replace(/\r/g,"\n").split("\n");
      log_state.partial=lines.pop() || "";
      const fragment=document.createDocumentFragment();
      if(log_state.partial){log_state.partial_node=ansi_line(log_state.partial);fragment.appendChild(log_state.partial_node);}
      for(const line of lines.reverse())fragment.appendChild(ansi_line(line));
      output.prepend(fragment);
      while(output.childElementCount>5000)output.lastElementChild.remove();
      if(!log_state.user_scrolled)output.scrollTop=0;
    }
    async function refresh_log(){
      if(document.visibilityState==="hidden" || pane.hidden || !app.current_run_id)return;
      if(log_state.run_id!==app.current_run_id)reset_log();
      if(log_state.controller)return;
      const run_id=app.current_run_id,controller=new AbortController();log_state.controller=controller;
      const deadline=setTimeout(()=>controller.abort(),15000);
      try{
        const query=new URLSearchParams({run:run_id,max_bytes:"262144"});
        if(log_state.offset!==null)query.set("offset",String(log_state.offset));
        const payload=await fetch_json(`/api/log?${query}`,{signal:controller.signal});
        if(controller.signal.aborted || run_id!==app.current_run_id || log_state.controller!==controller)return;
        if(!payload.available){status.textContent="No matching local console log was found for this run.";return;}
        if(payload.reset || payload.text)ingest_log(payload.text,Boolean(payload.reset || log_state.path && log_state.path!==payload.path));
        log_state.path=payload.path;log_state.offset=Number(payload.end);status.textContent=`${payload.path} · latest first`;
      }catch(error){if(!controller.signal.aborted && run_id===app.current_run_id)status.textContent=`Log read failed: ${error.message}`;}
      finally{clearTimeout(deadline);if(log_state.controller===controller)log_state.controller=null;}
    }
    const apply_tab_before=local_apply_detail_tab;
    local_apply_detail_tab=function(...args){
      const result=apply_tab_before.apply(this,args);pane.hidden=local_active_detail_tab!=="logs" || !app.current_run_id;
      if(!pane.hidden){by_id("run_blank_detail_pane").hidden=true;void refresh_log();}else log_state.controller?.abort();
      return result;
    };
    const heading_before=render_run_heading;
    render_run_heading=function(...args){const result=heading_before.apply(this,args);if(log_state.run_id!==app.current_run_id){reset_log();void refresh_log();}return result;};
    const style=document.createElement("style");style.textContent=`
      .run-logs-pane { display:flex; flex:1 1 auto; flex-direction:column; min-height:0; overflow:hidden; background:#000; color:#e6e6e6; }
      .run-logs-pane[hidden] { display:none !important; }
      .local-log-toolbar { flex:0 0 38px; display:flex; justify-content:space-between; align-items:center; padding:0 12px; background:#f3f4f5; color:#59616b; }
      #local_log_status { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; font-size:11px; }
      .local-log-toolbar > div { flex:none; white-space:nowrap; }
      #local_log_output { overflow:auto; flex:1 1 auto; min-height:0; padding:10px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; line-height:1.4; }
      .local-log-line { white-space:pre-wrap; overflow-wrap:anywhere; min-height:1.4em; }
    `;document.head.appendChild(style);
    setInterval(()=>void refresh_log(),1000);
    document.addEventListener("visibilitychange",()=>{if(document.visibilityState==="visible")void refresh_log();else log_state.controller?.abort();});
    local_apply_detail_tab();
    window.instra_run_logs={refresh_log,ingest_log,reset_log};
  },300);
});
// ^^^ THOG
