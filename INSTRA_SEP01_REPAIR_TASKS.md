# INSTRA September 1 Regression Repair - Tasks

Base: e53eb348ee2e863e873646f533cc011eaa5402b0
Branch: initialisation_baselining

- [x] Diagnose redundant refresh/redraw work, stale range responses, and deferred-render completion faults consistent with the report.
- [x] Move z after show overlapping range with a 12px left margin, shared by normal and magnified headers.
- [x] Make initial values select step zero and visibly indicate its selection; explicitly report when no step-zero snapshot was recorded.
- [x] Pass all 17 Node regression programs, three dependency-light Python tests, JavaScript syntax checks, and git diff --check.
- [x] Prepare reviewed files for GitHub connector publication and the download stanza. Completion gate: verify the published branch SHA before the final response.

Real Firefox/Plotly acceptance script: tests/instra_sep01_repair_browser.js. Not validated here: Firefox content processes are blocked by the environment's user-namespace sandbox restriction before a page can load. No sandbox bypass attempted. No measured real-browser speedup or reproduction of the user's machine is claimed.

Delivery: the Git commit containing this file on initialisation_baselining; the final response gives the connector-verified SHA. Restart Instra and hard-refresh after pulling. If further investigation is needed, use the supplied browser acceptance script in the normal project environment and the screenshot's three-run Workspace scenario.

Preserve the unrelated docs/PLASTIC_V056_PROGRESS.md modification. No parallel edits.
