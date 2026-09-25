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
  const pair = (container, key, value) => { append(container, "dt", key); append(container, "dd", value ?? "—"); };
  const button = (container, label, action, disabled = false) => {
    const control = append(container, "button", label);
    control.type = "button";
    control.disabled = disabled;
    control.addEventListener("click", action);
    return control;
  };
  async function json_response(url, options) {
    const response = await fetch(url, options);
    const raw = await response.text();
    let value;
    try { value = JSON.parse(raw); }
    catch { throw new Error(`HTTP ${response.status}: ${raw.trim().slice(0, 160) || "invalid JSON response"}`); }
    if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
    return value;
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
  async function run_action(name, host_id, args = {}) {
    message(`${name.replaceAll("_", " ")}…`);
    try {
      const result = await action(name, host_id, args);
      message(`${name.replaceAll("_", " ")} completed`);
      await refresh();
      return result;
    } catch (error) {
      // vvv THOG password-only SSH requires a fresh credential for each manual attempt
      const auth_host_id = error.failed_host_id || host_id;
      const multi_host_action = ["designate_master", "release_master", "settings"].includes(name);
      const supplied = multi_host_action ? args.passwords?.[auth_host_id] : args.password;
      if (error.category === "authentication" && auth_host_id && !supplied) {
        const host = snapshot?.hosts.find(item => item.thog_host_id === auth_host_id);
        const password = window.prompt(`SSH password for ${host?.address || auth_host_id} (this attempt only):`);
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
    if (!visible) return;
    try {
      snapshot = await json_response("/api/network");
      if (!snapshot.hosts.some(host => host.thog_host_id === selected_id)) selected_id = snapshot.local_id;
      render();
    } catch (error) { message(error.message); }
  }
  function render() {
    const list = element("network_host_list");
    list.replaceChildren();
    element("network_restart_mode").value = snapshot.restart_mode;
    if (document.activeElement !== element("network_retry_interval")) element("network_retry_interval").value = snapshot.retry_interval;
    for (const host of snapshot.hosts) {
      const row = button(list, "", () => { selected_id = host.thog_host_id; render(); });
      row.className = `network-host-row${host.thog_host_id === selected_id ? " active" : ""}`;
      append(row, "strong", `${host.display_name}${snapshot.master_id === host.thog_host_id ? " · Runner Master" : ""}`);
      append(row, "span", host.thog_host_id);
      append(row, "small", `${host.state} · ${host.last_discovered?.gpus?.length || 0} GPUs · ${host.monitoring_enabled ? "monitoring" : "no monitoring"} / ${host.execution_enabled ? "execution" : "no execution"}`);
    }
    const host = snapshot.hosts.find(item => item.thog_host_id === selected_id);
    if (host) render_detail(host);
  }
  function render_detail(host) {
    const discovery = host.last_discovered || {};
    const detail = element("network_detail");
    detail.replaceChildren();
    element("network_host_title").textContent = host.display_name;
    element("network_host_state").textContent = host.state;
    for (const tab of element("network_tabs").querySelectorAll("button")) tab.classList.toggle("active", tab.dataset.networkTab === selected_tab);
    const stale = host.state !== "available" ? " (stale)" : "";
    if (selected_tab === "overview") {
      const data = append(detail, "dl");
      pair(data, "Host ID", host.thog_host_id); pair(data, "SSH destination", host.address);
      pair(data, "Resolved IP", host.resolved_ip || discovery.resolved_ip);
      pair(data, "Remote hostname", discovery.hostname); pair(data, "SSH user / port", `${host.ssh_user || "SSH default"} / ${host.ssh_port}`);
      pair(data, "Authentication", host.authentication_mode); pair(data, "Agent", host.state);
      pair(data, "Last successful discovery", date_text(host.last_success) + stale);
      pair(data, "Discovery age", age_text(host.last_success)); pair(data, "Last contact", date_text(host.last_contact));
      pair(data, "Instra installation", discovery.instra?.root); pair(data, "Instra version", discovery.instra?.version);
      pair(data, "Instra process", discovery.instra?.running ? "Running" : "Not running at last discovery");
      pair(data, "THOG installation", discovery.thog?.root); pair(data, "THOG version", discovery.thog?.version);
      pair(data, "Operating system", discovery.os);
      if (host.latest_error) append(detail, "p", `Latest failure (${host.latest_error.category}): ${host.latest_error.message}`);
      const switches = append(detail, "div", undefined, "toggles");
      for (const [key, label] of [["monitoring_enabled", "Monitoring enabled"], ["execution_enabled", "Execution enabled"]]) {
        const wrapper = append(switches, "label");
        const check = append(wrapper, "input");
        check.type = "checkbox"; check.checked = host[key];
        check.addEventListener("change", () => run_action("update", host.thog_host_id, {[key]:check.checked}).catch(() => { check.checked = !check.checked; }));
        append(wrapper, "span", ` ${label}`);
      }
      const actions = append(detail, "div", undefined, "actions");
      button(actions, "Refresh discovery", () => run_action("discover", host.thog_host_id).catch(() => {}));
      button(actions, "Start Instra", () => run_action("start_instra", host.thog_host_id).catch(() => {}));
      button(actions, "Restart Instra", () => run_action("restart_instra", host.thog_host_id).catch(() => {}));
      if (snapshot.master_id === host.thog_host_id) {
        button(actions, snapshot.release_pending ? "Retry Runner Master release" : "Release Runner Master",
          () => run_action("release_master", host.thog_host_id).catch(() => {}));
      } else if (host.local && !snapshot.master_id) {
        button(actions, "Designate Runner Master", () => run_action("designate_master", host.thog_host_id).catch(() => {}));
      }
      if (!host.local) button(actions, "Remove", () => {
        if (confirm(`Remove ${host.display_name} from Instra configuration? Runs and files are retained.`)) run_action("remove", host.thog_host_id).catch(() => {});
      });
    } else if (selected_tab === "monitoring") {
      const data = append(detail, "dl");
      pair(data, "Monitoring", host.monitoring_enabled ? "Enabled" : "Disabled");
      pair(data, "Instra logs root", (discovery.instra_logs_root || "—") + stale);
      pair(data, "W&B root", (discovery.wandb_root || "—") + stale);
      const status = host.monitoring_status || {};
      pair(data, "Refresh interval", status.refresh_interval ?? "—"); pair(data, "Acquisition", status.activity ?? "—");
      pair(data, "Last acquisition", date_text(status.last_success)); pair(data, "Data age", age_text(status.last_success));
      pair(data, "Latest acquisition failure", status.latest_error || "—");
      // vvv THOG configure each monitoring instance's background check interval in existing host details
      const interval_label = append(detail, "label", "Check interval (seconds) ");
      const interval_input = append(interval_label, "input");
      interval_input.type = "number"; interval_input.min = "2"; interval_input.max = "300"; interval_input.step = "1";
      interval_input.value = String(status.refresh_interval || 5);
      interval_input.disabled = !host.monitoring_enabled;
      interval_input.addEventListener("change", () => run_action("monitor_settings", host.thog_host_id,
        {refresh_interval:Number(interval_input.value)}).catch(() => { interval_input.value = String(status.refresh_interval || 5); }));
      // ^^^ THOG
      button(detail, "Refresh run data", () => run_action("monitor_refresh", host.thog_host_id).catch(() => {}), !host.monitoring_enabled);
    } else if (selected_tab === "profiles") {
      const profiles = discovery.execution_profiles || [];
      if (!profiles.length) append(detail, "p", "No execution profiles discovered" + stale);
      for (const profile of profiles) {
        const card = append(detail, "section", undefined, "network-card"); append(card, "h3", profile.profile_key);
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
  setInterval(() => { if (visible) refresh(); }, 5000);
})();
// ^^^ THOG
