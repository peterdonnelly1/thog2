// vvv THOG
"use strict";

/* Existing NCU artifacts may predate the final documented class thresholds.
 * Normalize them in-memory so rerunning the profiler is unnecessary. */
(function install_processing_compatibility_classification_patch() {
  const corrected_class = row => {
    const pair = Boolean(row?.pair_can_co_reside);
    const one_main = Number(row?.premat_blocks_with_one_main_block || 0);
    const full_main = Number(row?.premat_blocks_with_full_main_residency || 0);
    if (!pair) return "RED";
    if (full_main >= 1) return "GREEN";
    if (one_main <= 1) return "ORANGE";
    return "YELLOW";
  };

  const normalize_source = source => {
    if (Array.isArray(source)) {
      return source.map(row => ({...row, compatibility_class: corrected_class(row)}));
    }
    if (source && Array.isArray(source.rows)) {
      return {...source, rows: source.rows.map(row => ({...row, compatibility_class: corrected_class(row)}))};
    }
    return source;
  };

  processing_gpu_compatibility_rank = function(value) {
    return {RED: 0, ORANGE: 1, YELLOW: 2, GREEN: 3}[String(value || "").toUpperCase()] ?? 99;
  };

  const processing_render_before_final_classification = processing_render;
  processing_render = async function(payload, trace_available) {
    if (!payload?.premat_compatibility) {
      return processing_render_before_final_classification(payload, trace_available);
    }
    const normalized = {
      ...payload,
      premat_compatibility: normalize_source(payload.premat_compatibility),
    };
    return processing_render_before_final_classification(normalized, trace_available);
  };
})();
// ^^^ THOG
