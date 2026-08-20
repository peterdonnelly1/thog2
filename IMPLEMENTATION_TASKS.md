# Implementation Tasks

## A. Decision algorithms

- [x] Add standalone Sen decision algorithm with configurable absolute slope threshold.
- [x] Add standalone Kendall decision algorithm with configurable absolute tau threshold.
- [x] Preserve both existing combined Sen-Kendall algorithms unchanged.
- [x] Add `jump_to_lowest_loss` bulldozer algorithm with configurable minimum percentage improvement.
- [x] Make bulldozer semantics and ignored controls explicit at startup.
- [x] Persist, resume, audit, console, help, and wrapper support for all new controls.
- [x] Add boundary, window, threshold, tie, feasibility, and resume regression tests.

## B. DENSE weight charts

- [x] Reuse the six existing THOG depth-wise weight chart families for DENSE.
- [x] Plot corresponding materialised DENSE weights at integer layer sample numbers.
- [x] Use unconnected cross markers.
- [x] Preserve coordinate selection, history colouring, hover metadata, destinations, and row limits.
- [x] Add local and W&B regression tests.

## C. Heatmap settings and data semantics

- [x] Remove `instrumentation__delta_loss_v_layer_heatmap_linear` as a public CLI option.
- [x] Add per-heatmap `probe count`, default 100.
- [x] Add fixed-from-zero versus rolling probe-count mode.
- [x] Add y display mode: probe rows or every optimizer step; labels always show optimizer steps.
- [x] Move heatmap-specific global settings into the heatmap gear panel.
- [x] Preserve compatible persisted/local data behavior.

## D. Heatmap visuals and event annotations

- [x] Increase x/y axis font sizes.
- [x] Permanently render committed decision bricks white with black text.
- [x] Render the next row's L axis label bold blue after a count change.
- [x] Show `update brake on` above the top x-axis.
- [x] Add a top absolute-layer axis aligned to the relative-offset axis.
- [x] Reposition brick-height slider between the mode control and gear.
- [x] Remove blank protrusions above the x-axis.
- [x] Round in-brick loss labels to three decimals.
- [x] Raise the legend title slightly.
- [x] Rename heatmap and remove `discrete cells` subtitle text.
- [x] Show active and reverted sampling-chaos-bump header messages in pale blue.
- [x] Default to percentage mode and label absolute mode `|abs|`.

## E. Verification and delivery

- [x] Run focused tests after each implementation stage.
- [x] Run the broad relevant regression suite.
- [x] Run static/syntax/browser-asset checks.
- [x] Review diff for unrelated changes and configuration compatibility.
- [x] Commit and publish the completed branch.

## F. Instra regression repair and multi-run Workspace

- [x] Restore per-weight-chart linear/signed-log controls and lighten DENSE cross strokes.
- [x] Rename the synthetic `coefficients` group to `weights` without changing stored chart-family names.
- [x] Restore the weight-chart top axis/title and make DENSE train/val group eligibility data-driven.
- [x] Suppress the heatmap group for DENSE runs.
- [x] Make all heatmap brick labels use one font family, size, mode and percentage format.
- [x] Rename the heatmap to `Heatmap - True Loss vs Counterfactual Layer Count Loss`.
- [x] Add a true top absolute-layer axis while retaining the relative-offset axis.
- [x] Remove blank heatmap protrusions and raise the colour-key title.
- [x] Render chaos-bump and update-brake state reliably from persisted per-row metadata.
- [x] Persist probe/step row mode on Apply and apply it to the main chart.
- [x] Use the CLI salmon colour for `update brake on`.
- [x] Add a left-rail Workspace view that overlays corresponding charts for visible runs.
- [x] Exclude heatmaps from Workspace and preserve single-run navigation.
- [x] Add source, server, persistence and multi-run figure regression tests.
- [ ] Run focused and broad regression suites, publish, and verify CI.
