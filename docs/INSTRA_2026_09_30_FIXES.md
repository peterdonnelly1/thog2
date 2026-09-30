# Instra fixes — 30 September 2026

Implemented on `distributed`, based on `465b1f93cf83cf5aa3a52604f33217ee203c00d7`.

## Outcomes

| Item | Result |
| --- | --- |
| 1 | Run menu offers deletion of all chart databases belonging to a Grid, with confirmation and active-Grid checks in both the UI and server. The warning says to Stop the Grid first. Existing acquired-remote-run protection is preserved: a Grid containing acquired copies must be deleted on its producing host. Checkpoints, training logs and W&B data are retained. |
| 2 | Bulk deletion uses one HTTP request, populates the identity cache once, clears successful selections immediately and reports individual failures. |
| 3 | Removed default buttons for Recipe label and the overall Eligible hosts and GPUs placement. Individual GPU power-cap defaults remain available. |
| 4 | Category default buttons are immediately to the left of their corresponding inputs. |
| 5 | History Grid labels and status are normal weight: green for every run completed, dark orange for some completed, red for none completed. |
| 6 | Grid central hues are persisted in browser storage. New hues maximize separation from all retained allocations. Individual members use shades of their Grid's hue. Reordering the catalogue or adding a Grid preserves existing central hues. Finite colour space still limits separation with very large Grid histories. |
| 7 | Main navigation has one selection owner driven by the active view. |
| 8 | Grid buttons say Rename. |
| 9 | Search is directly above the category buttons without its former section borders. |
| 10 | Estimates use hours, minutes and seconds as appropriate; 13687 seconds becomes 3h 48m 7s. |
| 11 | Progress and preview tables place Host and GPU immediately after Run ID. |
| 12 | Fixed a reproduced Plotly exception when an axis title was a string, eliminated repeated unchanged throughput relayouts, bounded previously unbounded chart requests and paused hidden chart polling. Chromium and Firefox each remained responsive during a five-minute 12-run Multiview soak. The specific Scruffy hardware lockup has not been directly reproduced or verified here. |
| 13 | Parameter hover help includes enumerated choices or the accepted value domain. Profiling, Recipe label, GPU selection, power cap and Search also explain their options. Seven missing catalogue choice lists now validate before launch. |
| 13b | Invalid parameter choices and syntax immediately set pure RGB(255,0,0) borders. Recipe label and GPU power-cap inputs also validate while typing. |
| 14 | Stored caps on unselected/unavailable GPUs no longer block an eligible placement; unticked hosts are skipped during placement preflight. A blank laptop cap with unavailable readback uses the existing driver policy without the privileged helper. Explicit unsupported caps produce a clear instruction to leave the field blank. A capability probe upgrades the verified local Node Agent on dashboard restart, terminating only its PID and retaining detached training children. Real Scruffy GPU execution remains unverified. |
| 15 | Enabled Premat category fields have a pale blue background. |
| 16 | Mode transitions clear old chart groups synchronously, discard stale asynchronous throughput responses, maximize loss when returning to Runs and show only the selected run's throughput curve. Loss is placed first. Navigation is synchronized after view changes. |

## Regression verification

- 172 Python tests and 10 subtests passed across Runner, Network, monitoring, local charts, live loss, processing pairs, earlier enhancement suites and time axes.
- All 39 JavaScript regression/smoke suites passed, including earlier September fixes.
- The complete dashboard passed Chromium and Firefox UI tests: options, immediate red validation, default controls, Search position, Premat colour, History outcomes, Progress columns and exclusive navigation.
- Each browser passed 15 rapid Multiview/Runs transitions with deliberately delayed chart HTTP responses, single-run versus 12-run throughput, maximized loss and zero uncaught browser exceptions.
- Each browser passed a five-minute Multiview soak. A 100 ms event-loop probe progressed throughout. This is an isolated fixture result, not a Scruffy hardware soak.
- Hidden Runner charts stopped making chart polling requests and resumed on return.
- Actual synthetic Grid deletion verified active warnings before confirmation, confirmation for terminal Grids, four-run removal in one request, and checkbox cleanup. Actual selected-run deletion removed three runs in one request.
- HTTP regression rejected active Grid deletion before touching its three databases, then deleted them after the Grid became terminal.
- Laptop-like unavailable power readbacks passed helper parsing and mocked reserve/launch/release without calling the privileged power helper or spawning real training.
- Launcher tests verify upgrade of old power-policy agents and continued acceptance of current agents; the recorded termination is the agent PID, not a training process group.
- Changed Python and JavaScript files passed syntax checks and `git diff --check`.

Browser fixture requests for absent optional `processing/update_timing.json` return expected 404s. The fixture has no timing capture. No tests connected to Scruffy or Dreedle, started real GPU jobs or interrupted any running sweep.

## Reproducing browser checks

Use an isolated `INSTRA_STATE_DIR`, CPU-compatible Python test dependencies, Node's `jsdom`, and Playwright with Chromium and Firefox installed. From the repository root:

```sh
INSTRA_STATE_DIR=/tmp/instra-regression-state python -m tests.instra_sep30_fixture /tmp/instra-browser-fixture 8765
```

In a second terminal:

```sh
INSTRA_SOAK_SECONDS=300 node tests/instra_sep30_browser_regression.js
```

The final browser phase deliberately deletes seven synthetic run databases. Start a fresh fixture directory before repeating the suite.

## Updating hosts

Pull `distributed` and restart the Scruffy dashboard normally. Its local agent upgrade is automatic. On Scruffy, leave the GPU power cap blank when the driver reports unavailable configurable power limits. Desktop caps continue to use validated write/readback and restoration. Updating or restarting Dreedle is unnecessary while its current sweep is running.
