// vvv THOG
"use strict";

const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const base_path = path.join(root, "sheet", "local_dashboard_assets", "dashboard_processing.js");
const resource_path = path.join(root, "sheet", "local_dashboard_assets", "dashboard_processing_resource_attribution.js");
const base = fs.readFileSync(base_path, "utf8");
const resource = fs.readFileSync(resource_path, "utf8");

new Function(`${base}\n${resource}`); // Parse the exact concatenation served by run_thog2_local_dashboard.py.

const required_fragments = [
  "Stream resource pressure",
  "processing_stream_resources",
  "MAIN_PREMAT_OVERLAP",
  "MIXED_SEQUENTIAL",
  "OTHER_OR_UNKNOWN",
  "processing_render_timeline_before_resource_attribution",
  "schema_version",
  "processing_resource_link_time_axes",
  "main_idle_intervals",
];
for (const fragment of required_fragments) {
  if (!resource.includes(fragment)) throw new Error(`missing Processing resource attribution fragment: ${fragment}`);
}

if (!resource.includes("return processing_render_timeline_before_resource_attribution(payload)")) {
  throw new Error("legacy Processing timeline fallback is not preserved");
}
if (!resource.includes("processing_download_stream_resources")) {
  throw new Error("stream resource CSV download is not exposed");
}
if (!resource.includes("processing_view.resource_axis_sync")) {
  throw new Error("linked timeline/resource x-axis recursion guard is missing");
}

console.log("processing resource attribution dashboard regression: ok");
// ^^^ THOG
