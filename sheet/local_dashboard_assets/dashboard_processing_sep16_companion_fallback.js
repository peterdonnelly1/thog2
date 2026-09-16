// vvv THOG
"use strict";

/*
 * NCU companion selection is now authoritative on the server. The browser used
 * to rediscover companions by probing candidate runs one-by-one, which made
 * navigation path-dependent and could add multi-second delays when several
 * closer NCU attempts lacked usable compatibility output.
 *
 * Keep this historical layer as a tiny payload handoff because later Processing
 * overlays consult companion_enriched_payload, but do not perform any searching.
 */
(function install_processing_sep16_companion_handoff() {
  const processing_render_before_companion_handoff = processing_render;
  processing_render = async function(payload, trace_available) {
    processing_view.companion_enriched_payload = payload;
    return processing_render_before_companion_handoff(payload, trace_available);
  };
})();
// ^^^ THOG
