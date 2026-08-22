// vvv THOG
"use strict";

// When one weight chart is maximized, the other five cards are display:none but
// the normal live refresh path still Plotly.react()ed all six. Keep only the
// newest skipped job per hidden weight chart and render it once when that card
// becomes visible again. This reduces fullscreen weight refresh work from six
// Plotly redraws to one without losing the latest figure on restore/switch.
window.addEventListener("load", () => {
  setTimeout(() => {
    const weight_chart_set = new Set([...depth_weight_chart_names]);
    const pending_by_chart = new Map();
    const stats = {skipped: 0, flushed: 0, rendered: 0};
    const base_render_plot_visibility = render_plot;

    const maximized_weight_chart = () => (
      weight_chart_set.has(app.maximized_chart) ? app.maximized_chart : null
    );

    const card_for_mount = mount => mount?.closest?.(".chart-card") || null;
    const card_is_visible = mount => {
      const card = card_for_mount(mount);
      return !card || card.offsetParent !== null;
    };

    const render_pending_job = async (chart_name, job) => {
      if (!job || job.run_id !== app.current_run_id) return false;
      const maximized = maximized_weight_chart();
      if (maximized && maximized !== chart_name) return false;
      if (!card_is_visible(job.mount)) return false;
      pending_by_chart.delete(chart_name);
      stats.flushed += 1;
      await base_render_plot_visibility(job.mount, job.figure, chart_name);
      return true;
    };

    const flush_visible_pending = async () => {
      for (const [chart_name, job] of [...pending_by_chart.entries()]) {
        try {
          await render_pending_job(chart_name, job);
        } catch (error) {
          show_toast(`Deferred weight chart refresh failed: ${error.message}`);
        }
      }
    };

    render_plot = function(mount, figure, chart_name) {
      const maximized = maximized_weight_chart();
      if (
        weight_chart_set.has(chart_name)
        && maximized
        && maximized !== chart_name
      ) {
        pending_by_chart.set(chart_name, {
          mount,
          figure,
          run_id: app.current_run_id,
        });
        stats.skipped += 1;
        return Promise.resolve();
      }
      pending_by_chart.delete(chart_name);
      stats.rendered += 1;
      return base_render_plot_visibility(mount, figure, chart_name);
    };

    const queue_flush = () => {
      requestAnimationFrame(() => requestAnimationFrame(() => {
        flush_visible_pending();
      }));
    };

    const base_toggle_maximized_chart_visibility = toggle_maximized_chart;
    toggle_maximized_chart = function(chart_name) {
      const result = base_toggle_maximized_chart_visibility(chart_name);
      queue_flush();
      return result;
    };

    const base_restore_maximized_chart_visibility = restore_maximized_chart;
    restore_maximized_chart = function() {
      const result = base_restore_maximized_chart_visibility();
      queue_flush();
      return result;
    };

    const base_toggle_chart_group_visibility = toggle_chart_group;
    toggle_chart_group = function(button) {
      const result = base_toggle_chart_group_visibility(button);
      queue_flush();
      return result;
    };

    const base_select_run_visibility = select_run;
    select_run = function(run_id, options = {}) {
      if (run_id !== app.current_run_id) pending_by_chart.clear();
      return base_select_run_visibility(run_id, options);
    };

    window.__instra_render_visibility_performance = {
      pending_count: () => pending_by_chart.size,
      flush: flush_visible_pending,
      stats: () => ({...stats, pending: pending_by_chart.size}),
    };
  }, 1800);
});
// ^^^ THOG
