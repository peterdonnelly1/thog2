// vvv THOG
"use strict";

(function install_processing_sep16_companion_fallback() {
  const compatibility_rows = payload => {
    const source = payload?.premat_compatibility;
    if (Array.isArray(source)) return source;
    if (source && Array.isArray(source.rows)) return source.rows;
    return [];
  };

  function encoded_suffix(run) {
    const artifact = String(run?.artifact_name || run?.run_name || "");
    const marker = artifact.indexOf("___");
    return marker >= 0 ? artifact.slice(marker + 3) : "";
  }

  function companion_candidates() {
    const selected = typeof current_run === "function" ? current_run() : null;
    if (!selected) return [];
    const selected_id = String(run_identifier(selected));
    const suffix = encoded_suffix(selected);
    const host = String(selected.host_label || "");
    if (!suffix) return [];
    return (app.runs || [])
      .filter(run => (
        String(run_identifier(run)) !== selected_id
        && encoded_suffix(run) === suffix
        && String(run.host_label || "") === host
      ))
      .sort((left, right) => {
        const left_time = Date.parse(left.created_at || left.updated_at || "") || 0;
        const right_time = Date.parse(right.created_at || right.updated_at || "") || 0;
        return right_time - left_time;
      });
  }

  async function browser_discovered_companion(payload, trace_available) {
    if (!trace_available || compatibility_rows(payload).length) return payload;
    for (const run of companion_candidates()) {
      const run_id = String(run_identifier(run));
      try {
        const response = await fetch_json(`/api/processing?run=${encodeURIComponent(run_id)}`);
        const data = response?.data;
        if (!data || !compatibility_rows(data).length) continue;
        return {
          ...payload,
          premat_compatibility:data.premat_compatibility,
          premat_hard_constraints:[...(data.premat_hard_constraints || [])],
          premat_compatibility_files:{...(data.premat_compatibility_files || {})},
          premat_compatibility_source:{
            dashboard_run_id:run_id,
            artifact_name:String(run.artifact_name || run.run_name || run_id),
            created_at:String(run.created_at || ""),
            host_label:String(run.host_label || ""),
            pair_key:`${run.host_label || ""}|${encoded_suffix(run)}`,
            discovery:"browser_fallback",
          },
        };
      } catch (_error) {
        // Try the next exact-config candidate.
      }
    }
    return payload;
  }

  const processing_render_before_companion_fallback = processing_render;
  processing_render = async function(payload, trace_available) {
    const enriched = await browser_discovered_companion(payload, Boolean(trace_available));
    processing_view.companion_enriched_payload = enriched;
    return processing_render_before_companion_fallback(enriched, trace_available);
  };
})();
// ^^^ THOG
