# Instra run controls, loss settings and recipe editing

Branch: `residuals_compression`. Parent: `e8934cec495fd60a0d06059661d95b630733aafc`.

## Requested outcomes

| # | Outcome |
|---|---|
| 1 | Removed the per-run red bins; the toolbar bin and existing single-run menu remain available. |
| 2 | The toolbar bin uses a darker gray outline whenever a checkbox selection exists, including a mixed selection that cannot be deleted. |
| 3 | Main-grid run colour patches are square. |
| 4 | Space toggles the focused run's checkbox without scrolling or changing run focus. Polling also preserves row/control focus. |
| 5 | Bulk confirmation and tooltip say “Delete Instra data for NN selected runs”. |
| 6 | The toolbar can force-delete all selected remote copies after confirmation. Mixed/local/unknown selections cannot reach that force-delete path. |
| 7 | Loss settings render a live preview in Runs and Multiview, including maximized-chart contexts. |
| 8 | “Reset ranges to Auto” clears X minimum/maximum and Y minimum/maximum while retaining other preferences. |
| 9 | Dense preset/status text is bold in the main table and Runner views. Preset metadata changes invalidate the table cache. |
| 10 | Progress uses the heading `preset`. |
| 11 | Progress has `est. end` after `end`, using live progress or the existing completed-run exemplar estimate. Unavailable estimates show an em dash. |
| 12 | Progress omits Profiling; History retains that information. |
| 13 | Each run has a dedicated 3x3 rectangular grid switch before its eye. Grid switches activate/deactivate the whole cohort; eyes override one run. Preferences persist and new members inherit the grid state. |
| 14 | Loss settings offer fine, light-gray minor grid lines at one tenth of resolved major tick spacing; zoom updates the intervals. |
| 15 | `!-` immediately removes front-curve bolding; `!+` restores/pins it. Explicit emphasis survives polling and z-order changes. |
| 16 | Each Recipe has Edit before Rename. Explicit Edit saves in place; the left panel expands to preserve the name button's width. |
| 17 | Runner wall clocks use two spaces between date and time, preserved by the rendered table. |
| 18 | Rename changes only the existing saved Recipe's name, preserving its identity, parameters and launched Grid snapshots. |

## Regression protection

- The former heatmap patch yielded ownership of the toolbar bin to the current
  handler. It otherwise attached a second confirmation/deletion path and could
  overwrite current selection state.
- Native checkbox handlers update toolbar state after the checkbox's own change
  listener. A capture-phase microtask could run before that listener on a mouse
  event, leaving the bin one selection behind.
- The demand runtime recognizes the settings preview independently of the chart
  viewport and maximized card. Ordinary hidden charts remain deferred.
- Minor-grid relayout listeners are reattached after Plotly purges without adding
  duplicate listeners.
- Explicit Recipe Edit and Rename preserve `recipe_id`, `created_at` and
  `last_grid_id`; launched Grid snapshots remain unchanged. Ordinary Save retains
  the existing numbered-copy workflow.
- Force deletion validates the entire selection on the backend before queuing
  any work. It removes local acquired copies through the existing background
  service and preserves producer authority and normal reacquisition behavior.

## Validation

The broad Python run passed 429 tests and 32 subtests. It also reported eight
inherited failures reproduced on the unchanged parent, which passed 421 tests:
six September CLI spelling subtests fail from the existing global argparse
import-order conflict; the September chart-route fixture has an inherited
response-shape expectation; and the geometry console test expects obsolete
report wording. The training CLI and geometry behavior were not changed.

The focused API/Recipe/estimate tests pass all nine cases. All 41 non-browser
JavaScript regression programs pass. Python compilation, JavaScript syntax and
`git diff --check` pass.

The existing October UI regression suite passes in Chromium and Firefox,
including table drag/resize persistence, colours, logs, Recipe Reset, dated
History, visible-curve JSON/BIFF8 exports, Processing zoom and 30 delayed
navigation cycles per browser. The distributed-deletion/emphasis browser suite
also passes in both browsers. The actual Width HTTP/browser acceptance passes.

Final new-control and live-soak results are recorded in
`evidence/instra_run_controls_20261006.json`. The new browser suite covers all
18 items with actual Plotly rendering, a real Recipe store and production API
dispatch tests. Chromium exercises settings footer buttons by native mouse;
Firefox exercises those buttons by native Enter because this host drops some
footer mouse events even on a plain HTML test page. Other tested Firefox controls
use native mouse/keyboard input. No programmatic click dispatch substitutes for
those settings button activations.

The 32-run live comparison measures steady state after startup requests settle,
preserves zoom, checks hidden-chart idleness and tests bounded rendering with
32 x 40,000 raw throughput points and complete exports. The browser fixture and
CPU tests exercise UI, storage and API behavior; they do not run remote GPU jobs.
