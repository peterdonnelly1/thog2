// vvv THOG present host configuration, discovery and diagnostics without SSH in the browser
"use strict";
(function install_networks() {
  const element = id => document.getElementById(id);
  const view = element("network_view");
  if (!view) return;
  let snapshot = null;
  let selected_id = null;
  let selected_tab = "overview";
  let visible = false;
  let pending_fingerprint = null;
  let refresh_watch_until = 0;
  let refresh_in_flight = false;
  const message = value => { element("network_message").textContent = value || ""; };
  const date_text = value => value ? new Date(value).toLocaleString() : "—";
  const age_text = value => value ? `${Math.max(0, Math.round((Date.now() - Date.parse(value)) / 60000))} min` : "—";
  const append = (parent, tag, content, class_name) => {
    const child = document.createElement(tag);
    if (content !== undefined) child.textContent = String(content ?? "—");
    if (class_name) child.className = class_name;
    parent.appendChild(child);
    return child;
  };
  const pair = (container, key, value, value_class) => {
    append(container, "dt", key);
    append(container, "dd", value ?? "—", value_class);
  };
  const button = (container, label, action, disabled = false) => {
    const control = append(container, "button", label);
    control.type = "button";
    control.disabled = disabled;
    control.addEventListener("click", action);
    return control;
  };
  async function json_response(url, options) {
    const abort = new AbortController();
    const deadline = setTimeout(() => abort.abort(), 20000);
    try {
      const response = await fetch(url, {...options, signal:abort.signal});
      const raw = await response.text();
      let value;
      try { value = JSON.parse(raw); }
      catch { throw new Error(`HTTP ${response.status}: ${raw.trim().slice(0, 160) || "invalid JSON response"}`); }
      if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
      return value;
    } finally { clearTimeout(deadline); }
  }
  async function action(name, host_id = null, args = {}) {
    const result = await json_response("/api/network/action", {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({action:name, host_id, args})});
    const job_id = result.job_id;
    for (;;) {
      await new Promise(resolve => setTimeout(resolve, 220));
      const job = await json_response(`/api/network/job?id=${encodeURIComponent(job_id)}`);
      if (job.status === "working") continue;
      if (job.status === "error") {
        const failure = new Error(job.error);
        failure.category = job.category;
        failure.failed_host_id = job.failed_host_id;
        throw failure;
      }
      return job.result;
    }
  }
  // vvv THOG keep one-attempt SSH passwords masked unless the user explicitly shows them
  function request_password(host) {
    const dialog = element("network_auth_dialog");
    const form = element("network_auth_form");
    const input = element("network_auth_password");
    const cancel = element("network_auth_cancel");
    element("network_auth_message").textContent = `SSH key or certificate authentication failed for ${host?.ssh_user || "your SSH user"}@${host?.address || "this host"}. Check that Instra can use your existing SSH credential. A password is optional and is sent for this attempt only.`;
    input.value = "";
    input.type = "password";
    element("network_auth_eye").setAttribute("aria-label", "Show password");
    element("network_auth_eye").setAttribute("aria-pressed", "false");
    return new Promise(resolve => {
      const finish = value => {
        form.onsubmit = null; cancel.onclick = null; dialog.oncancel = null;
        if (dialog.open) dialog.close();
        input.value = ""; input.type = "password";
        resolve(value);
      };
      form.onsubmit = event => { event.preventDefault(); finish(input.value || null); };
      cancel.onclick = () => finish(null);
      dialog.oncancel = event => { event.preventDefault(); finish(null); };
      dialog.showModal();
      input.focus();
    });
  }
  function toggle_password(input, control) {
    const visible = input.type === "password";
    input.type = visible ? "text" : "password";
    control.setAttribute("aria-label", visible ? "Hide password" : "Show password");
    control.setAttribute("aria-pressed", String(visible));
  }
  element("network_auth_eye").addEventListener("click", () => toggle_password(element("network_auth_password"), element("network_auth_eye")));
  element("network_password_eye").addEventListener("click", () => toggle_password(form_password(), element("network_password_eye")));
  function form_password() { return element("network_add_form").elements.password; }
  // ^^^ THOG
  async function run_action(name, host_id, args = {}) {
    message(name === "monitor_refresh" ? "Requesting run data refresh…" : `${name.replaceAll("_", " ")}…`);
    try {
      const result = await action(name, host_id, args);
      if (name === "monitor_refresh") refresh_watch_until = Date.now() + 30000;
      message(name === "monitor_refresh" ? "Run data refresh queued; acquisition status is updating below" : `${name.replaceAll("_", " ")} completed`);
      await refresh();
      return result;
    } catch (error) {
      // vvv THOG password-only SSH requires a fresh credential for each manual attempt
      const auth_host_id = error.failed_host_id || host_id;
      const multi_host_action = ["designate_master", "release_master", "settings"].includes(name);
      const supplied = multi_host_action ? args.passwords?.[auth_host_id] : args.password;
      if (error.category === "authentication" && auth_host_id && !supplied) {
        const host = snapshot?.hosts.find(item => item.thog_host_id === auth_host_id);
        const password = await request_password(host);
        if (password) {
          const retry_args = multi_host_action ? {...args, passwords:{...args.passwords, [auth_host_id]:password}} : {...args, password};
          return run_action(name, host_id, retry_args);
        }
      }
      // ^^^ THOG
      message(`${error.category || "Network"}: ${error.message}`);
      throw error;
    }
  }
  async function refresh() {
    if (!visible || refresh_in_flight) return;
    refresh_in_flight = true;
    try {
      snapshot = await json_response("/api/network");
      if (!snapshot.hosts.some(host => host.thog_host_id === selected_id)) selected_id = snapshot.local_id;
      render();
    } catch (error) { message(error.message); }
    finally { refresh_in_flight = false; }
  }
  function render() {
    const list = element("network_host_list");
    list.replaceChildren();
    element("network_restart_mode").value = snapshot.restart_mode;
    if (document.activeElement !== element("network_retry_interval")) element("network_retry_interval").value = snapshot.retry_interval;
    for (const host of snapshot.hosts) {
      // vvv THOG place removal beside its host; selection and deletion are separate buttons
      const row = append(list, "div", undefined, `network-host-row${host.thog_host_id === selected_id ? " active" : ""}`);
      const select = button(row, "", () => { selected_id = host.thog_host_id; render(); });
      select.className = "network-host-select";
      append(select, "strong", `${host.display_name}${snapshot.master_id === host.thog_host_id ? " · Runner Master" : ""}`);
      append(select, "span", host.thog_host_id);
      append(select, "small", `${host.state} · ${host.last_discovered?.gpus?.length || 0} GPUs · ${host.monitoring_enabled ? "monitoring" : "no monitoring"} / ${host.execution_enabled ? "execution" : "no execution"}`,
        host.state === "discovering" ? "network-discovering" : "");
      if (!host.local) {
        const remove_button = button(row, "Remove", () => {
          if (snapshot.master_id) {
            message("Release Runner Master in the Execution tab before removing a participating host.");
            return;
          }
          if (confirm(`Remove ${host.display_name} from Instra configuration? Runs and files are retained.`)) {
            run_action("remove", host.thog_host_id).catch(() => {});
          }
        });
        remove_button.className = "network-host-remove";
        remove_button.setAttribute("aria-label", `Remove ${host.display_name}`);
        remove_button.title = snapshot.master_id ? "Release Runner Master in Execution before removing a participating host" : `Remove ${host.display_name}`;
      }
      // ^^^ THOG
    }
    const host = snapshot.hosts.find(item => item.thog_host_id === selected_id);
    // Keep the editor mounted while its user is typing; polling must not replace the input.
    const editing_interval = document.activeElement?.dataset?.networkCheckInterval === selected_id
      && selected_tab === "monitoring";
    if (host && !editing_interval) render_detail(host);
  }
  function render_detail(host) {
    const discovery = host.last_discovered || {};
    const detail = element("network_detail");
    detail.replaceChildren();
    element("network_host_title").textContent = host.display_name;
    const state_label = element("network_host_state");
    state_label.textContent = host.state;
    state_label.classList.toggle("network-discovering", host.state === "discovering");
    // vvv THOG expose host discovery above all detail tabs, including Monitoring
    const host_actions = element("network_host_actions");
    host_actions.replaceChildren();
    button(host_actions, "Refresh discovery", () => run_action("discover", host.thog_host_id).catch(() => {}));
    // ^^^ THOG
    for (const tab of element("network_tabs").querySelectorAll("button")) tab.classList.toggle("active", tab.dataset.networkTab === selected_tab);
    const stale = host.state !== "available" ? " (stale)" : "";
    if (selected_tab === "overview") {
      const data = append(detail, "dl");
      pair(data, "Host ID", host.thog_host_id); pair(data, "SSH destination", host.address);
      pair(data, "Resolved IP", host.resolved_ip || discovery.resolved_ip);
      pair(data, "Remote hostname", discovery.hostname); pair(data, "SSH user / port", `${host.ssh_user || "SSH default"} / ${host.ssh_port}`);
      pair(data, "Authentication", host.authentication_mode);
      pair(data, "Agent", host.state, host.state === "available" ? "network-healthy" :
        host.state === "discovering" ? "network-discovering" : "network-unhealthy");
      pair(data, "Last successful discovery", date_text(host.last_success) + stale);
      pair(data, "Discovery age", age_text(host.last_success)); pair(data, "Last contact", date_text(host.last_contact));
      pair(data, "Dashboard installation", discovery.instra?.root); pair(data, "Dashboard version", discovery.instra?.version);
      pair(data, "Dashboard process", discovery.instra?.running ? "running" : "not running at last discovery",
        discovery.instra?.running ? "network-healthy" : "network-unhealthy");
      pair(data, "THOG installation", discovery.thog?.root); pair(data, "THOG version", discovery.thog?.version);
      pair(data, "Operating system", discovery.os);
      if (host.latest_error) append(detail, "p", `Latest failure (${host.latest_error.category}): ${host.latest_error.message}`);
      if (host.local) {
        append(detail, "p", "This is the dashboard currently open in your browser. Its node agent is a separate background service.", "network-muted");
      } else {
        append(detail, "p", "The Instra process is this host's dashboard. Its separate node agent reports host status even when that dashboard is stopped.", "network-muted");
        const actions = append(detail, "div", undefined, "actions");
        button(actions, `Start dashboard on ${host.display_name}`, () => run_action("start_instra", host.thog_host_id).catch(() => {}));
        button(actions, `Restart dashboard on ${host.display_name}`, () => run_action("restart_instra", host.thog_host_id).catch(() => {}));
      }
    } else if (selected_tab === "monitoring") {
      toggle(detail, host, "monitoring_enabled", `Enable other thog hosts to monitor runs on ${host.display_name}`);
      const data = append(detail, "dl");
      pair(data, "Instra logs root", (discovery.instra_logs_root || "—") + stale);
      pair(data, "W&B root", (discovery.wandb_root || "—") + stale);
      const status = host.monitoring_status || {};
      pair(data, "Refresh interval", status.refresh_interval ?? "—"); pair(data, "Acquisition", status.activity ?? "—");
      pair(data, "Last acquisition", date_text(status.last_success)); pair(data, "Data age", age_text(status.last_success));
      pair(data, "Latest acquisition failure", status.latest_error || "—", status.latest_error ? "network-unhealthy" : "");
      // vvv THOG configure each monitoring instance's background check interval in existing host details
      const interval_label = append(detail, "label", "Check interval (seconds) ");
      const interval_input = append(interval_label, "input");
      interval_input.type = "number"; interval_input.min = "2"; interval_input.max = "300"; interval_input.step = "1";
      interval_input.value = String(status.refresh_interval || 5);
      interval_input.disabled = !host.monitoring_enabled;
      interval_input.dataset.networkCheckInterval = host.thog_host_id;
      interval_input.addEventListener("change", () => run_action("monitor_settings", host.thog_host_id,
        {refresh_interval:Number(interval_input.value)}).catch(() => { interval_input.value = String(status.refresh_interval || 5); }));
      interval_input.addEventListener("blur", () => { if (visible) refresh(); });
      // ^^^ THOG
      button(detail, "Manually refresh run data now", () => run_action("monitor_refresh", host.thog_host_id).catch(() => {}), !host.monitoring_enabled);
    } else if (selected_tab === "profiles") {
      toggle(detail, host, "execution_enabled", `Enable other thog hosts to execute runs on ${host.display_name}`);
      const master_row = append(detail, "div", undefined, "toggles");
      const master_label = append(master_row, "label");
      const master_check = append(master_label, "input");
      master_check.type = "checkbox";
      master_check.checked = snapshot.master_id === host.thog_host_id;
      master_check.disabled = !host.local || Boolean(snapshot.master_id && snapshot.master_id !== host.thog_host_id);
      master_check.addEventListener("change", () => {
        const command = master_check.checked ? "designate_master" : "release_master";
        run_action(command, host.thog_host_id).catch(() => { master_check.checked = !master_check.checked; });
      });
      append(master_label, "span", " Runner Master");
      if (!host.local) append(detail, "p", "Designate Runner Master on the local host's Execution tab.", "network-muted");
      if (snapshot.release_pending && master_check.checked) append(detail, "p", "Runner Master release pending; deselect to retry.", "network-muted");
      const profiles = discovery.execution_profiles || [];
      if (!profiles.length) append(detail, "p", "No execution profiles discovered" + stale);
      for (const profile of profiles) {
        const card = append(detail, "section", undefined, "network-card");
        if (profile.profile_key !== "current") append(card, "h3", profile.profile_key);
        const data = append(card, "dl"); pair(data, "Profile ID", profile.execution_profile_id);
        pair(data, "Location", profile.location); pair(data, "Python", profile.python);
        pair(data, "Shell entry", profile.shell_entry); pair(data, "THOG entry", profile.thog_entry);
      }
    } else if (selected_tab === "gpus") {
      pair(append(detail, "dl"), "CUDA version", discovery.cuda_version);
      const gpus = discovery.gpus || [];
      if (!gpus.length) append(detail, "p", "No CUDA GPUs discovered" + stale);
      for (const gpu of gpus) {
        const card = append(detail, "section", undefined, "network-card"); append(card, "h3", `CUDA ${gpu.ordinal}: ${gpu.model}`);
        const data = append(card, "dl"); pair(data, "GPU ID", gpu.gpu_id);
        pair(data, "NVIDIA UUID", gpu.uuid); pair(data, "Memory (MiB)", gpu.memory_mib); pair(data, "Driver", gpu.driver);
        pair(data, "Compute processes at discovery", (gpu.compute_pids || []).join(", ") || "None");
      }
    } else if (selected_tab === "logs") {
      append(detail, "p", "Material Network, Monitoring and Runner events for this host.");
      json_response(`/api/network/events?host_id=${encodeURIComponent(host.thog_host_id)}`).then(value => {
        if (selected_id !== host.thog_host_id || selected_tab !== "logs") return;
        for (const event of value.events) append(detail, "p", `${date_text(event.time)} · ${event.source} · ${event.operation} · ${event.outcome}: ${event.message}`, "network-card");
      }).catch(error => message(error.message));
    }
  }
  function toggle(detail, host, key, label) {
    const switches = append(detail, "div", undefined, "toggles");
    const wrapper = append(switches, "label");
    const check = append(wrapper, "input");
    check.type = "checkbox";
    check.checked = host[key];
    check.addEventListener("change", () => run_action("update", host.thog_host_id, {[key]:check.checked}).catch(() => { check.checked = !check.checked; }));
    append(wrapper, "span", ` ${label}`);
  }
  function show(which) {
    visible = which === "networks";
    view.hidden = !visible;
    element("runner_view").hidden = which !== "runner";
    for (const id of ["workspace_nav", "runs_nav", "networks_nav", "runner_nav", "settings_nav"]) {
      element(id)?.classList.toggle("selected", id === `${which}_nav`);
    }
    if (which === "networks") { element("breadcrumb_leaf").textContent = "Networks"; refresh(); }
    if (which === "runner") element("breadcrumb_leaf").textContent = "Runner";
  }
  element("networks_nav").addEventListener("click", () => show("networks"));
  element("runner_nav").addEventListener("click", () => show("runner"));
  for (const id of ["runs_nav", "workspace_nav", "settings_nav"]) element(id)?.addEventListener("click", () => {
    view.hidden = true; element("runner_view").hidden = true; visible = false;
    element("networks_nav").classList.remove("selected"); element("runner_nav").classList.remove("selected");
  });
  for (const tab of element("network_tabs").querySelectorAll("button")) tab.addEventListener("click", () => {
    selected_tab = tab.dataset.networkTab; render();
  });
  element("network_restart_mode").addEventListener("change", event => run_action("settings", null, {restart_mode:event.target.value}).catch(() => {}));
  element("network_retry_interval").addEventListener("change", event => run_action("settings", null, {retry_interval:Number(event.target.value)}).catch(() => {}));
  const dialog = element("network_add_dialog");
  const form = element("network_add_form");
  element("network_add_button").addEventListener("click", () => {
    form.reset(); pending_fingerprint = null;
    form_password().type = "password";
    element("network_password_eye").setAttribute("aria-label", "Show password");
    element("network_password_eye").setAttribute("aria-pressed", "false");
    element("network_password_label").hidden = true;
    element("network_fingerprint").hidden = true;
    element("network_add_error").textContent = "";
    element("network_add_submit").textContent = "Connect";
    dialog.showModal();
  });
  element("network_add_cancel").addEventListener("click", () => dialog.close());
  element("network_password_button").addEventListener("click", () => { element("network_password_label").hidden = false; form.elements.password.focus(); });
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const args = {address:form.elements.address.value.trim(), ssh_user:form.elements.ssh_user.value.trim(),
      ssh_port:Number(form.elements.ssh_port.value), password:form.elements.password.value || undefined};
    const error_box = element("network_add_error");
    error_box.textContent = ""; element("network_add_submit").disabled = true;
    try {
      if (!pending_fingerprint) {
        const identity = await action("prepare_host", null, args);
        if (!identity.known) {
          pending_fingerprint = identity.fingerprint;
          element("network_fingerprint").textContent = `Verify this SSH host fingerprint independently, then explicitly accept: ${pending_fingerprint}`;
          element("network_fingerprint").hidden = false;
          element("network_add_submit").textContent = "Accept fingerprint and connect";
          return;
        }
      }
      args.fingerprint = pending_fingerprint;
      const added = await run_action("add", null, args);
      selected_id = added.thog_host_id;
      dialog.close();
      await refresh();
    } catch (error) {
      error_box.textContent = `${error.category || "Network"}: ${error.message}`;
      if (error.category === "authentication") element("network_password_label").hidden = false;
    } finally { form.elements.password.value = ""; element("network_add_submit").disabled = false; }
  });
  let refresh_tick = 0;
  setInterval(() => { if (visible && (Date.now() < refresh_watch_until || ++refresh_tick % 5 === 0)) refresh(); }, 1000);
})();
// ^^^ THOG
