// vvv THOG keep discovery and remote removal accessible from Monitoring and other host tabs
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.events = {};
    this.dataset = {};
    const classes = new Set();
    this.classList = {contains:name => classes.has(name), add:name => classes.add(name),
      remove:name => classes.delete(name), toggle:(name, force) => {
        if (force === undefined ? !classes.has(name) : force) classes.add(name); else classes.delete(name);
      }};
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren() { this.children = []; }
  addEventListener(name, handler) { this.events[name] = handler; }
  setAttribute(name, value) { this[name] = value; }
  showModal() { this.open = true; }
  close() { this.open = false; }
  focus() {}
  querySelectorAll() { return this.children; }
  click() { return this.events.click?.({preventDefault() {}}); }
}

async function main() {
  const elements = new Map();
  const element = name => {
    if (!elements.has(name)) elements.set(name, new Element());
    return elements.get(name);
  };
  const tabs = element("network_tabs");
  for (const name of ["overview", "monitoring", "profiles", "gpus", "logs"]) {
    const tab = new Element("button");
    tab.dataset.networkTab = name;
    tabs.appendChild(tab);
  }
  const calls = [];
  const document = {getElementById:element, createElement:tag => new Element(tag), activeElement:null};
  let fail_auth = false;
  const hosts = [
    {thog_host_id:"thog_host.scruffy", display_name:"scruffy", local:true, state:"available",
      monitoring_enabled:true, execution_enabled:false, last_discovered:{}},
    {thog_host_id:"thog_host.dreedle", display_name:"dreedle", local:false, state:"available",
      monitoring_enabled:true, execution_enabled:false, monitoring_status:{refresh_interval:5},
      last_discovered:{instra_logs_root:"/logs", wandb_root:"/wandb"}},
  ];
  const snapshot = {hosts, local_id:hosts[0].thog_host_id, master_id:null,
    restart_mode:"off", retry_interval:30};
  const response = value => ({ok:true, text:async () => JSON.stringify(value)});
  const context = {
    document,
    window: {prompt:() => { throw new Error("cleartext browser prompt was used"); }},
    confirm:() => true,
    setInterval() {},
    setTimeout:callback => { callback(); },
    clearTimeout() {}, AbortController,
    fetch:async (url, options) => {
      if (url === "/api/network") return response(snapshot);
      if (url === "/api/network/action") {
        calls.push(JSON.parse(options.body));
        if (calls.at(-1).action === "monitor_settings") {
          hosts.find(host => host.thog_host_id === calls.at(-1).host_id).monitoring_status.refresh_interval = calls.at(-1).args.refresh_interval;
        }
        return response({job_id:"done"});
      }
      if (url.startsWith("/api/network/job")) {
        if (fail_auth && calls.at(-1)?.action === "discover" && !calls.at(-1).args.password) {
          return response({status:"error", category:"authentication", error:"SSH authentication failed"});
        }
        return response({status:"done", result:{}});
      }
      if (url.startsWith("/api/network/events")) return response({events:[]});
      throw new Error(url);
    },
    Date, Number, String, console,
  };
  const source = fs.readFileSync(path.join(__dirname, "../sheet/local_dashboard_assets/dashboard_networks.js"), "utf8");
  vm.runInNewContext(source, context);
  element("networks_nav").click();
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(element("network_host_actions").children.map(child => child.textContent), ["Refresh discovery"]);

  element("network_host_list").children[1].children[0].click();
  tabs.children[1].click();
  assert.equal(element("network_detail").children[0].children[0].children[1].textContent,
    " Enable other thog hosts to monitor runs on dreedle");
  assert.equal(element("network_detail").children.at(-1).textContent, "Manually refresh run data now");
  const interval = element("network_detail").children[2].children[0];
  interval.value = "17";
  document.activeElement = interval;
  element("networks_nav").click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(element("network_detail").children[2].children[0], interval, "polling replaced the focused interval editor");
  assert.equal(interval.value, "17");
  interval.events.change();
  document.activeElement = null;
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(element("network_detail").children[2].children[0].value, "17");
  assert.equal(calls.at(-1).action, "monitor_settings");
  hosts[1].state = "discovering";
  element("networks_nav").click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(element("network_host_state").classList.contains("network-discovering"), true);
  hosts[1].state = "available";
  const actions = element("network_host_actions").children;
  assert.deepEqual(actions.map(child => child.textContent), ["Refresh discovery"]);
  const remove_button = element("network_host_list").children[1].children[1];
  assert.equal(remove_button.textContent, "Remove");
  assert.equal(element("network_host_list").children[0].children.length, 1);
  actions[0].click();
  remove_button.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(calls.slice(-2).map(call => [call.action, call.host_id]), [
    ["discover", "thog_host.dreedle"], ["remove", "thog_host.dreedle"],
  ]);
  snapshot.master_id = "thog_host.scruffy";
  element("networks_nav").click();
  await new Promise(resolve => setImmediate(resolve));
  element("network_host_list").children[1].children[0].click();
  tabs.children[0].click();
  assert.equal(element("network_host_list").children[1].children[1].disabled, false);
  const prior_calls = calls.length;
  element("network_host_list").children[1].children[1].click();
  assert.equal(calls.length, prior_calls);
  assert.match(element("network_message").textContent, /Release Runner Master/);
  element("network_host_list").children[0].children[0].click();
  tabs.children[2].click();
  const master_check = element("network_detail").children[1].children[0].children[0];
  assert.equal(master_check.checked, true);
  assert.equal(element("network_detail").children[0].children[0].children[1].textContent,
    " Enable other thog hosts to execute runs on scruffy");
  assert.equal(element("network_host_actions").children[0].textContent, "Refresh discovery");
  fail_auth = true;
  element("network_host_actions").children[0].click();
  await new Promise(resolve => setImmediate(resolve));
  const auth_dialog = element("network_auth_dialog");
  const password = element("network_auth_password");
  assert.equal(auth_dialog.open, true);
  assert.equal(password.type, "password");
  element("network_auth_eye").click();
  assert.equal(password.type, "text");
  element("network_auth_eye").click();
  assert.equal(password.type, "password");
  password.value = "one-attempt-secret";
  element("network_auth_form").onsubmit({preventDefault() {}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(calls.at(-1).args.password, "one-attempt-secret");
  assert.equal(password.value, "");
  assert.equal(password.type, "password");
  assert.equal(auth_dialog.open, false);
  console.log("PASS Networks discovery, list-row removal, master guard and masked SSH retry");
}

main().catch(error => { console.error(error); process.exitCode = 1; });
// ^^^ THOG
