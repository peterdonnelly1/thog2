// vvv THOG
"use strict";

// Keep the heatmap newest-first by actual optimiser step, and prevent an
// in-flight train.log response from one run being ingested after another run
// has become selected.
window.addEventListener("load", () => {
  setTimeout(() => {
    let latest_heatmap_step = null;

    const first_step_from_row = row => {
      if (!Array.isArray(row)) return null;
      for (const cell of row) {
        if (!Array.isArray(cell) || cell.length < 1) continue;
        const numeric = Number(cell[0]);
        if (Number.isFinite(numeric)) return numeric;
      }
      return null;
    };

    const base_transpose_heatmap_newest_first = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      base_transpose_heatmap_newest_first(prepared);
      const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (!heatmap_trace) return;

      const coordinates = Array.isArray(heatmap_trace.y)
        ? heatmap_trace.y.map(Number)
        : [];
      const customdata = Array.isArray(heatmap_trace.customdata)
        ? heatmap_trace.customdata
        : [];
      const finite_coordinates = coordinates.filter(Number.isFinite);
      if (!finite_coordinates.length) return;

      let newest_index = -1;
      let newest_step = -Infinity;
      for (let index = 0; index < customdata.length; index += 1) {
        const step = first_step_from_row(customdata[index]);
        if (step !== null && step >= newest_step) {
          newest_step = step;
          newest_index = index;
        }
      }
      if (newest_index < 0) newest_index = coordinates.length - 1;
      const newest_coordinate = Number(coordinates[newest_index]);
      if (!Number.isFinite(newest_coordinate)) return;
      latest_heatmap_step = Number.isFinite(newest_step) ? newest_step : null;

      const minimum = Math.min(...finite_coordinates);
      const maximum = Math.max(...finite_coordinates);
      const sorted_unique = [...new Set(finite_coordinates)].sort((left, right) => left - right);
      let pitch = 1;
      for (let index = 1; index < sorted_unique.length; index += 1) {
        const difference = sorted_unique[index] - sorted_unique[index - 1];
        if (difference > 0) {
          pitch = difference;
          break;
        }
      }
      const lower_edge = minimum - pitch / 2;
      const upper_edge = maximum + pitch / 2;
      const newest_is_on_high_side = (
        Math.abs(newest_coordinate - maximum) <= Math.abs(newest_coordinate - minimum)
      );

      prepared.layout.yaxis = {
        ...(prepared.layout.yaxis || {}),
        // For a Plotly y-axis the second range endpoint is the top of the plot.
        // Choose the orientation from the actual newest-step coordinate rather
        // than assuming chronological coordinates are ascending or descending.
        range: newest_is_on_high_side
          ? [lower_edge, upper_edge]
          : [upper_edge, lower_edge],
        autorange: false,
      };
      prepared.layout.meta = {
        ...(prepared.layout.meta || {}),
        thog2_latest_heatmap_step: latest_heatmap_step,
      };
    };

    const heatmap_mount = by_id("heatmap_plot");
    const style_latest_tick = () => {
      if (!heatmap_mount) return;
      const tick_texts = [...heatmap_mount.querySelectorAll(".ytick text")];
      for (const text of tick_texts) {
        text.classList.remove("thog2-latest-y-tick");
      }
      if (!Number.isFinite(latest_heatmap_step)) return;
      const wanted = String(latest_heatmap_step);
      const match = tick_texts.find(text => String(text.textContent || "").trim() === wanted);
      if (match) match.classList.add("thog2-latest-y-tick");
    };
    if (heatmap_mount) {
      const observer = new MutationObserver(style_latest_tick);
      observer.observe(heatmap_mount, {childList: true, subtree: true, characterData: true});
      style_latest_tick();
    }

    const base_fetch_json_log_run_guard = fetch_json;
    fetch_json = async function(url, ...arguments_after_url) {
      const raw_url = String(url || "");
      let parsed = null;
      try {
        parsed = new URL(raw_url, window.location.origin);
      } catch (_error) {
        return base_fetch_json_log_run_guard(url, ...arguments_after_url);
      }
      if (parsed.pathname !== "/api/log") {
        return base_fetch_json_log_run_guard(url, ...arguments_after_url);
      }

      let requested_run = parsed.searchParams.get("run") || "";
      let request_url = `${parsed.pathname}${parsed.search}`;
      const maximum_bytes = parsed.searchParams.get("max_bytes") || String(2 * 1024 * 1024);

      // A run switch can occur while the HTTP request is in flight. Never hand
      // that stale payload back to the log viewer. Reissue from offset zero for
      // the newly selected run and keep following rapid switches until stable.
      for (let attempt = 0; attempt < 5; attempt += 1) {
        const payload = await base_fetch_json_log_run_guard(
          request_url,
          ...arguments_after_url,
        );
        const current_run = String(app.current_run_id || "");
        if (!current_run || current_run === requested_run) return payload;

        requested_run = current_run;
        const fresh = new URL("/api/log", window.location.origin);
        fresh.searchParams.set("run", requested_run);
        fresh.searchParams.set("max_bytes", maximum_bytes);
        request_url = `${fresh.pathname}${fresh.search}`;
      }

      // Extremely rapid repeated switching: return no stale text and let the
      // normal one-second poll retry the now-current run.
      return {
        available: false,
        path: "",
        size: 0,
        start: 0,
        end: 0,
        reset: true,
        text: "",
      };
    };

    const base_select_run_log_reset = select_run;
    select_run = function(run_id, options = {}) {
      const changing_run = String(run_id || "") !== String(app.current_run_id || "");
      if (changing_run) {
        const output = by_id("local_log_output");
        if (output) {
          output.replaceChildren();
          output.scrollTop = 0;
        }
        const status = by_id("local_log_status");
        if (status) status.textContent = "Loading train.log for selected run…";
      }
      return base_select_run_log_reset(run_id, options);
    };

    const style = document.createElement("style");
    style.textContent = `
      /* The old selector assumed the newest tick was the final SVG tick. Once
         the heatmap can run either axis direction, style only the actual newest
         optimiser-step tick identified above. */
      .chart-card[data-chart="heatmap"] .ytick:last-of-type text:not(.thog2-latest-y-tick) {
        font-weight: 400 !important;
        transform: none !important;
      }
      .chart-card[data-chart="heatmap"] .ytick text.thog2-latest-y-tick {
        font-size: 15px !important;
        font-weight: 800 !important;
        fill: #171a1f !important;
        transform: translate(-7px, 7px) !important;
      }
    `;
    document.head.appendChild(style);

    if (app.figures?.heatmap && app.current_run_id) {
      queueMicrotask(async () => {
        const mount = by_id("heatmap_plot");
        if (mount) await render_plot(mount, app.figures.heatmap, "heatmap");
      });
    }
  }, 0);
});
// ^^^ THOG
