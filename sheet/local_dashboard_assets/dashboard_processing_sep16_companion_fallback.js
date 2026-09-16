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

  function artifact_time_ms(run) {
    const artifact = String(run?.artifact_name || run?.run_name || "");
    const match = artifact.match(/^(\d{2})(\d{2})(\d{2})-(\d{2})(\d{2})_/);
    if (match) {
      const [, yy, mm, dd, hh, min] = match;
      return Date.UTC(2000 + Number(yy), Number(mm) - 1, Number(dd), Number(hh), Number(min));
    }
    return Date.parse(run?.created_at || run?.updated_at || "") || 0;
  }

  function is_ncu_candidate(run) {
    const artifact = String(run?.artifact_name || run?.run_name || "").toUpperCase();
    return artifact.includes("_NCU_") || artifact.includes("NCU_PREMAT");
  }

  function strip_companion(payload) {
    const cleaned = {...payload};
    delete cleaned.premat_compatibility;
    delete cleaned.premat_hard_constraints;
    delete cleaned.premat_compatibility_files;
    delete cleaned.premat_compatibility_source;
    return cleaned;
  }

  function companion_candidates() {
    const selected = typeof current_run === "function" ? current_run() : null;
    if (!selected) return [];
    const selected_id = String(run_identifier(selected));
    const suffix = encoded_suffix(selected);
    const host = String(selected.host_label || "");
    const selected_time = artifact_time_ms(selected);
    if (!suffix) return [];
    return (app.runs || [])
      .filter(run => (
        String(run_identifier(run)) !== selected_id
        && is_ncu_candidate(run)
        && encoded_suffix(run) === suffix
        && String(run.host_label || "") === host
      ))
      .map(run => ({
        run,
        time:artifact_time_ms(run),
        distance:Math.abs(artifact_time_ms(run) - selected_time),
      }))
      .sort((left, right) => (
        left.distance - right.distance
        || right.time - left.time
      ));
  }

  async function browser_discovered_companion(payload, trace_available) {
    const base = strip_companion(payload);
    if (!trace_available) return base;

    const candidates = companion_candidates();
    const skipped = [];
    for (const candidate of candidates) {
      const run = candidate.run;
      const run_id = String(run_identifier(run));
      try {
        const response = await fetch_json(`/api/processing?run=${encodeURIComponent(run_id)}`);
        const data = response?.data;
        if (!data || !compatibility_rows(data).length) {
          skipped.push(String(run.artifact_name || run.run_name || run_id));
          continue;
        }
        return {
          ...base,
          premat_compatibility:data.premat_compatibility,
          premat_hard_constraints:[...(data.premat_hard_constraints || [])],
          premat_compatibility_files:{...(data.premat_compatibility_files || {})},
          premat_compatibility_source:{
            dashboard_run_id:run_id,
            artifact_name:String(run.artifact_name || run.run_name || run_id),
            created_at:String(run.created_at || ""),
            host_label:String(run.host_label || ""),
            pair_key:`${run.host_label || ""}|${encoded_suffix(run)}`,
            discovery:"browser_authoritative_nearest_viable",
            distance_minutes:candidate.distance / 60000.0,
            skipped_closer_invalid_count:skipped.length,
            skipped_closer_invalid_artifacts:[...skipped],
          },
        };
      } catch (_error) {
        skipped.push(String(run.artifact_name || run.run_name || run_id));
      }
    }
    return base;
  }

  const processing_render_before_companion_fallback = processing_render;
  processing_render = async function(payload, trace_available) {
    const enriched = await browser_discovered_companion(payload, Boolean(trace_available));
    processing_view.companion_enriched_payload = enriched;
    return processing_render_before_companion_fallback(enriched, trace_available);
  };
})();
// ^^^ THOG
