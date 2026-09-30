# Instra October 1: History, Grid controls and Downloads

Base: distributed at 5286b187a90d6df3a05e7a4d6a9e1d1572178504.

| Request | Result |
| --- | --- |
| 1 | History has aligned T and T_est columns. T measures wall time from first dispatch to terminal state, excluding initial queue wait. T_est displays the immutable launch-time interval. Missing historical timing or estimates are shown as unknown. |
| 2 | Manual run colours override automatic Grid shades and persist through reloads. The existing picker now accepts an automatic HSL shade on opening. |
| 3 | Opening an eye selects the Grid; closing an eye hides only that run. The existing grouping setting can still disable Grid selection. |
| 4 | Grid-wide deletion uses the same icon, red colour and menu role as individual deletion. |
| 6a | State follows the Grid button, before Rename. History uses full width; the remaining diagnostic column retains complete messages with a small right margin. |
| 6b | History Delete uses the existing dark red action colour. |
| 7 | Grid state summaries show completed/total runs. Failed and cancelled runs do not count as completed. |
| 8 | Every available export has View and Download controls, one shared preview and a visible display label. Per-Grid preview selections and scroll position survive polling; obsolete responses are cancelled. |
| 9 | Response headers and download attributes advertise Grid-prefixed names for every export. Stored legacy paths remain compatible. |
| 10 | Removed Scripts; both script forms remain in Downloads. |
| 11 | Changed the main Runs loss heading and its later UI override to loss. |

The controller persists first-dispatch and terminal timestamps, clears only the terminal timestamp on retry and leaves the upfront estimate unchanged. Legacy timing is reconstructed from attempt timestamps and final attempt duration when sufficient metadata exists. Parallel durations are not summed.

Validation on the final implementation:

- 160 Python tests and 4 subtests passed: Runner/Node Agent, Networks, Monitoring, pair probes, prior September 30 fixes, timing persistence, live loss, chart axes, weight ranges, heatmap windows, Premat and processing resource/NCU regressions.
- 40 JavaScript regression suites passed, covering existing chart, pairing, selection, colour and runtime ownership paths as well as new timing/HSL cases.
- New Chromium and Firefox acceptance tests passed: History column coordinates, complete diagnostics, elapsed/estimated values, state counts, actual bytes and suggested filenames for every download, delayed preview switching, polling retention, Grid switching, saved colours and eyes, single-run exclusions, removal of the hidden current-run curve, shared delete styling and narrow-viewport overflow.
- Prior feedback acceptance passed in Chromium and Firefox: category controls, saved eyes, progressive slow loss, cancellation and stale plastic cleanup.
- Prior navigation/deletion acceptance passed in Chromium and Firefox, including 30-second responsiveness soaks per browser and actual batch deletion/checkbox cleanup. One rapid-navigation run timed out during concurrent setup; the unchanged base and a subsequent isolated rerun passed. No chart/navigation production changes were needed.
- History and Downloads screenshots were inspected. JavaScript syntax, changed Python compilation and git whitespace checks passed.

No GPU training run was required for these dashboard/controller changes.
