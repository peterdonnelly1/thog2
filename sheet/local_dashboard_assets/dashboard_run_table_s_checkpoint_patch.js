// vvv THOG
"use strict";

// S is the established activation-checkpoint segment-size knob, not the Boolean
// enable flag.  Keep the column numeric when checkpointing is enabled and show
// "off" only when activation checkpointing itself is disabled.
window.addEventListener("load", () => {
  setTimeout(() => {
    const checkpoint_segment_text = run => {
      const configuration = run?.configuration && typeof run.configuration === "object"
        ? run.configuration
        : {};
      if (configuration.activation_checkpointing === false) return "off";
      const value = Number(configuration.checkpoint_segment_size);
      return Number.isInteger(value) && value > 0 ? String(value) : "—";
    };

    const repair_checkpoint_segment_cells = () => {
      for (const row of document.querySelectorAll(".runs-table tbody tr[data-run-id]")) {
        const run_id = String(row.dataset.runId || "");
        const run = app.runs.find(candidate => run_identifier(candidate) === run_id);
        const cell = row.querySelector('[data-instra-run-shape-cell="activation_checkpointing"]');
        if (!run || !cell) continue;
        cell.textContent = checkpoint_segment_text(run);
        cell.title = `activation checkpointing segment size: ${cell.textContent}`;
      }
      const header = document.querySelector('[data-instra-run-shape-header="activation_checkpointing"]');
      if (header) header.title = "activation checkpointing segment size";
    };

    const base_render_runs_checkpoint_segment = render_runs;
    render_runs = function() {
      const result = base_render_runs_checkpoint_segment();
      repair_checkpoint_segment_cells();
      return result;
    };

    repair_checkpoint_segment_cells();
  }, 1600);
});
// ^^^ THOG
