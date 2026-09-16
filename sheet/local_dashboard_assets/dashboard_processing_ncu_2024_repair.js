// vvv THOG
"use strict";

/*
 * Final Processing repair overlay for NCU 2024.3 and high-resolution INSTRA.
 *
 * - renders NCU compatibility data even when no NSYS processing_data.json exists;
 * - restores the Workspace rail icon that still exists in the DOM but lost its CSS;
 * - thickens MAIN/PREMAT execution bars in the GPU processing timeline.
 */

(function install_processing_ncu_2024_repair() {
  const style_id = "instra-processing-ncu-2024-repair-style";
  if (!document.getElementById(style_id)) {
    const style = document.createElement("style");
    style.id = style_id;
    style.textContent = `
      .workspace-icon {
        width: 22px;
        height: 18px;
        display: grid;
        grid-template-columns: 9px 9px;
        grid-template-rows: 8px 8px;
        gap: 2px;
      }
      .workspace-icon i {
        display: block;
        min-width: 0;
        min-height: 0;
        border: 1.5px solid currentColor;
        border-radius: 2px;
      }
      .workspace-icon i:first-child {
        grid-row: 1 / span 2;
      }
    `;
    document.head.appendChild(style);
  }

  async function thicken_processing_timeline_bars() {
    const mount = by_id("processing_timeline_plot");
    if (!mount || mount.dataset.plotReady !== "true" || !Array.isArray(mount.data)) return;
    const jobs = [];
    mount.data.forEach((trace, index) => {
      const name = String(trace?.name || "");
      const width = name.startsWith("PREMAT ") ? 12 : (name.startsWith("MAIN ") ? 10 : null);
      if (width === null) return;
      jobs.push(Plotly.restyle(mount, {"line.width": width}, [index]));
    });
    if (jobs.length) await Promise.all(jobs);
  }

  const processing_render_before_ncu_2024_repair = processing_render;
  processing_render = async function(payload, trace_available) {
    await processing_render_before_ncu_2024_repair(payload, trace_available);

    const compatibility_rows = (
      typeof processing_gpu_compatibility_rows === "function"
        ? processing_gpu_compatibility_rows(payload)
        : []
    );

    /*
     * NCU is intentionally not an NSYS temporal trace, so trace_available is
     * false.  The compatibility card is nevertheless a valid selected-run
     * Processing result and must be rendered from its own immutable artifacts.
     */
    if (
      !trace_available
      && compatibility_rows.length
      && typeof processing_resource_render === "function"
    ) {
      await processing_resource_render(payload);
      if (typeof processing_sync_visibility === "function") processing_sync_visibility();
    }

    await thicken_processing_timeline_bars();
  };
})();
// ^^^ THOG
