# Implementation Log

## 2026-08-20

- Started from local `depth_weight_curves_and_observational_probes` at `aacfb2a`, one local commit ahead of the configured remote-tracking reference.
- Confirmed the working tree was clean before creating these tracking files.
- Established staged implementation order: algorithms, DENSE charts, heatmap settings/data, visuals/events, comprehensive regression testing.
- Added PLASTIC v0.57 decision ownership with standalone `sen`, standalone `kendall`, and `jump_to_lowest_loss`.
- Added independent Sen-slope, Kendall-tau, and raw-loss improvement thresholds with CLI routing, help, persistence, identity, resume-mismatch checks, startup rows and audit replay.
- Standalone statistical modes retain full-radius stratified evidence, growth-side discount, raw-adjacent improvement veto, configured window, update brake and unit-step commits.
- Bulldozer mode ranks finite raw validation losses independently of the configured objective, ignores the statistical window and max-step, and retains update/warmup/CUDA safety gates. Same-batch mode is reduced to one-probe windows only for this algorithm.
- Added focused selector, threshold, brake, tie, direct-jump, validation, parser/persistence and wrapper regression tests. Python syntax, shell syntax and wrapper routing checks pass; the torch-dependent suite remains scheduled for the comprehensive test phase.
- Added DENSE snapshot capture for the same Q/K/V/attention-output/MLP-up/MLP-down chart families as THOG.
- DENSE snapshots read actual learned block weights at integer layer indices. Q/K/V correctly address the three row slices of nanoGPT's combined attention input projection.
- DENSE figures use unconnected cross markers, preserve deterministic scalar/head selection, accumulated-history age styling, hover metadata, local SQLite storage and W&B destinations, while THOG's continuous curves and executed-layer circles remain unchanged.
- Added DENSE family, projection-slice, marker, coordinate-lock and local-history regression tests; syntax and diff checks pass pending the dependency-backed suite.
- Removed the public heatmap linear-max-step argument and all capture-time truncation. Local heatmap storage now retains complete probe history; linear W&B publication and promoted probe progress are no longer capped by the retired field.
- Added server-side, cached heatmap windows with a validated 1–512 viewer probe count (default 100) and rolling/from-zero selection.
- Moved capture metadata, probe window, row mode, Δloss mode and colour-band controls into the heatmap gear. The old workspace-level heatmap settings section is retained only as hidden compatibility DOM for earlier composed patches.
- Added exact every-optimizer-step expansion with blank non-probe rows, while y-axis labels remain optimizer-step values in both row modes.
- Persisted committed decisions, update-brake state, selected counts and sampling-chaos-bump state alongside each local heatmap record, including backward-compatible defaults for existing SQLite rows.
- Added permanent white/black committed-decision cells, transition-blue L labelling, brake and chaos headers, a latest-L absolute top axis, larger axes, three-decimal cell loss labels, percent-default mode, `|abs|`, revised title/subtitle, adjusted key title and header control placement.
- Python and JavaScript syntax checks plus `git diff --check` pass after the heatmap data/UI stage; focused and broad behavioral suites remain in progress.
- Removed the retired heatmap capture-limit option from the remaining shell and lifecycle routes. Existing checkpoint-schema storage remains readable, but the field is ignored and is no longer accepted by either public CLI.
- Corrected final UI composition so the visible `|abs|`/`%` control is centred and percentage mode is the actual default; the hidden legacy button is no longer the positioning target.
- Added indexed 1–512-record SQLite window reads so a rolling live view does not decode the complete run on every probe refresh.
- Reviewed decision dispatch and fixed the v0.56-retired/unknown-name error path to avoid recursive delegation after runtime ownership is patched.
- Static verification passes: Python compileall, all dashboard JavaScript `node --check`, Bash syntax, and `git diff --check`.
- Focused wrapper routing suite passes (10/10). Pure direct-loss selector, DENSE chart source-contract, and Instra heatmap source-contract smoke checks pass.
- A 32-test cross-cutting shell regression selection has the same 12 failures and 5 dependency/baseline errors as the exact pre-change commit's 31-test selection. The failures are pre-existing missing Torch/runtime assets and legacy absent-wrapper expectations, not new differences.
- The local environment has no Torch, Plotly, pytest, or browser binary, so dependency-backed chart/controller tests cannot execute here. Added all new tests plus affected heatmap/dashboard suites to the branch's GitHub CPU workflow, which installs the required dependencies and runs on publish.
- Final staged diff is scoped to the three requested feature areas, their configuration/help surfaces, Instra presentation/data paths, tests, CI coverage and these handoff records. Prepared for commit and branch publication.
- Published the exact tested tree through the connected GitHub integration. The first dependency-backed focused run reported 5 failures among 143 focused tests (138 passed).
- Root-caused four failures to the process-level underscore alias normaliser rewriting the deliberately exact PLASTIC/chaos/instrumentation namespaces after the new late parser layer, and one to a chart test relying on the former W&B destination default.
- Updated `sitecustomize.py` to preserve all exact public double-underscore namespaces while retaining ordinary underscore aliases, added a direct regression, and made the W&B visibility test select its intended destination explicitly.
- The corrected focused CI suite passed. The exhaustive 211-module head-vs-base comparison then identified exactly three stale v0.55/v0.56 ownership/help assertions; all implementation tests passed and the other 23 non-passing modules reproduced equivalently on the base.
- Updated those legacy integration assertions to recognise v0.57 as the final selector owner and to validate the expanded six-algorithm/help surface, including the bulldozer objective override.

## 2026-08-20 — Instra regression repair and Workspace

- Began a fresh acceptance pass against the reported live UI rather than relying on the earlier checked implementation list.
- Preserved the user's unrelated `sheet/trainer_schedule.py` working-tree change and excluded it from this task.
- Identified the dashboard's composed patch/load order as the first dependency to resolve: several v0.57 behaviours exist in source files but are not all part of the served asset graph, while later geometry overrides can replace axis semantics.
- Established repair order: asset/data eligibility and setting persistence; one authoritative heatmap renderer; weight-chart presentation; multi-run Workspace; focused and broad regression verification.
- Added one final v0.58 browser owner after v0.57, repaired the invalid weight-scale DOM insertion, and retained internal `coefficients` identifiers while displaying the group as `weights`.
- Added multi-run Workspace aggregation across visible runs for both stored weight figures and locally scanned W&B metric groups. Heatmaps are excluded and DENSE heatmap groups are hidden.
- Rebuilt final heatmap axes and overlays from persisted row metadata: latest-L absolute top axis, uniform deduplicated labels, permanent white decision bricks, black L band, brake/chaos headers, protrusion removal, and Apply-time persistence.
- Added a Node semantic smoke test covering Workspace aggregation, DENSE eligibility, settings persistence, weight preparation, heatmap axes, decision labels, and brake/chaos state. JavaScript syntax, inline scripts, Python compilation, and scoped whitespace checks pass locally; dependency-complete Python and broad comparison suites remain delegated to branch CI.
- Verified the reported brake sequence: transition rows 76, 88, 100, 112 and 124 are unbraked; each preceding row is the last visibly braked row and each following row re-enters the post-transition brake window. No brake logic was changed.
- Root-caused the still-missing top ordinates/title to Plotly dropping an overlaid axis with no trace assigned to `x2`. Added one transparent, non-interactive anchor trace for every weight chart and the heatmap, and made the final figure-preparation owner restore the heatmap's absolute-layer title after older presentation wrappers run.
- Extended the Node semantic smoke test to require the `x2` anchor for both chart families, require exactly one heatmap anchor after repeated preparation, and verify restoration of the latest-L absolute title. All dashboard JavaScript syntax and semantic smoke checks pass locally.
- Removed the redundant Plotly-internal weight title in Workspace only; the card header remains the sole Workspace title owner, while single-run figures retain their detailed source title.
- Changed each DENSE optimizer-step/scalar trace from crosses alone to crosses joined by 0.45 px straight segments. Connectors never cross step, scalar, or run boundaries and THOG curve traces are unchanged.
- Extended source and browser-semantic tests for Workspace title ownership, single-run title retention, cross-marker preservation, straight connector width and run-colour propagation.
- Published the top-axis/Workspace-title/DENSE-connector tree; the remote branch records that exact tree as `b2d2691`.
- Refined DENSE history presentation so each recorded optimizer step owns one stable random-looking HSL colour and one legend row, while all selected scalar coordinates within that step share the colour and retain their scalar identity in hover metadata.
- Reduced DENSE crosses to a 4–6 px range with a 0.35 px cross stroke; retained 0.45 px straight connectors within each scalar/step only. Removed all `oldest` / `newest` title and legend language.
- Added per-weight-chart Instra display settings for `Step count` and `Step window` (`Rolling` or `From zero`). Windowing reads explicit DENSE optimizer-update metadata first, retains the existing THOG `U…` fallback, and sorts updates numerically; it never uses PLASTIC probe indices.
- Added semantic Node coverage for DENSE Workspace colour/legend behavior, marker styling, and earliest/latest optimizer-step windows. Dashboard JavaScript syntax, semantic smoke, Python compilation and scoped whitespace checks pass locally. The dependency-complete chart suites remain covered by the branch workflow.

## 2026-08-21 — Workspace weight-history controls

- Reconstructed the requested continuation from the active repository because the prior-chat lookup returned no usable turn state; no partial implementation was present.
- Preserved the unrelated user modification in `sheet/trainer_schedule.py` and excluded it from all task edits.
- Removed all Workspace weight legends while retaining full trace names for hover identity and leaving ordinary single-run legends unchanged.
- Added a `Current weights only` Display Preferences toggle for all six weight charts. Its state is scoped independently to each run/chart and to each Workspace chart.
- Implemented Workspace current-only filtering per visible run, so runs at different optimizer steps each retain their own newest snapshot rather than being compared against one global maximum step.
- Verified both DENSE and THOG update identification, multiple scalars at one DENSE step, per-run Workspace maxima, run/Workspace setting isolation, absent Workspace legends, JavaScript syntax across every dashboard asset, Python source compilation, scoped whitespace and the v0.58 semantic smoke test.
- Full dependency-backed Python tests cannot run in this sandbox because Plotly and pytest are not installed; the affected source-contract and chart suites remain in the branch CI workflow.
