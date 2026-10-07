# Instra and Chebyshev update, 8 October 2026

Branch: `residuals_compression`. Baseline: `629bdd1b3c43e2e4bc91fd31b863facce58383b3`.

The Chebyshev construction now samples ascending roots of T_N and normalises its columns directly. The shared DEPTH, WIDTH and compressed THOGOPT history paths use this construction. The old QR construction is replaced; its version is rejected rather than offered as a compatibility mode. The construction is equivalent to the orthonormal DCT-II basis up to ordering and signs. Conditioning is corrected; training quality still requires experiments.

Instra skips the complete table rendering pipeline when its displayed data is unchanged, avoids copying immutable chart histories during workspace merging, uses a run lookup map while polishing rows, and obtains snapshot counts and endpoints with indexed SQLite operations. Forced local deletion persists across polling and restart, and can be resumed from settings.

Duration estimates use comparable execution configurations, iteration counts, observed startup cost, GPU/power/profiling policy and the actual GPU queue. Unsupported extrapolations remain unknown. The retained `max_parallel` recipe field does not limit the current Runner dispatcher, so estimates use the available selected GPUs, matching execution. External GPU queue waits remain outside the estimate.

| Item | Outcome |
| --- | --- |
| 1 | Reduced repeated table, chart-history and catalogue work; sustained 12-run browser polling passed. |
| 2 | Workload-scaled duration estimates with compatible histories, startup fitting and GPU scheduling. |
| 3 | Chebyshev root nodes and direct normalisation; WIDTH 512 and 1024 at D=1024 accepted; curve and metadata paths updated. |
| 4 | Hover help documents valid choices, registered compressor versions, parameterised syntax and Chebyshev/DCT-II equivalence. |
| 5 | Rename action reads "Save to New Name". |
| 6 | WIDTH message explains combined compression using `--select-depth`; `depth, width` requests separate trials in the current CLI. |
| 7 | Requested state names are green or red; blocked and stopping retain their neutral colour. |
| 8 | Removed the Multiview information panel from global settings. |
| 9 | Global settings use an organised responsive dialog; heatmap controls are expandable. |
| 10 | Pending producer chart data remains accessible in a collapsed disclosure. |
| 11 | Automatic run colour has Very light, Light, Medium and Dark options, defaulting to Light. Explicit run/grid choices are preserved. |
| 12 | Runs table displays "width". |
| 13 | Adjacent L/P and D/r pairs; D/r are dark blue and r remains lowercase. Existing layouts migrate once and retain later user rearrangement. |
| 14 | Forced local deletions stay excluded across polls and restart; settings offer Resume monitoring. |
| 15 | Range-only Auto reset is above the X/Y axis controls with grey spacing. |
| 16 | Loss minor grids are fine and pale, at one tenth of each major interval; verified after zoom and reload. |

Validation evidence is in `evidence/instra_oct08_validation.json`. The initial complete Python run reported 1,902 passes, 484 passing subtests, 27 skips and 97 failures. All 77 ordinary non-browser failing cases reproduce on the unchanged baseline; ten failing subtests also reproduce there. Ten legacy Firefox/Selenium cases cannot start their backend under the execution environment's Unix-socket restriction. Separate Playwright acceptance suites pass in Chromium and Firefox.

The final broad regression pass reported 1,902 passes, 480 passing subtests and 27 skips, with the 91 verified baseline or environment-blocked cases explicitly deselected. There were no failures. Additional targeted optimizer/root checks reported 49 passes and four GPU skips; snapshot and curve checks reported 27 passes and 18 passing subtests. All 43 JavaScript/HTTP regression suites passed, including automatic recovery from repeated telemetry timeouts.

The 180-second Chromium soak exercised 12 live loss curves, 172 health requests and 126 loss redraws. The hidden chart stayed at one draw. Peak heap was 78.6 MiB; retained heap increased from 14.3 to 16.0 MiB. The largest measured polling pause was 865 ms, with no page errors. These are fixture measurements, not a long-duration measurement on dreedle or scruffy.

Update both hosts to this branch, restart the Instra backend on each, and hard-refresh the browser with Ctrl+Shift+R. New training uses the replacement basis. Existing model weights are not migrated.
