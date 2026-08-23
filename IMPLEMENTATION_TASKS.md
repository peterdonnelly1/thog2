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
- [x] Use cross markers with faint straight segments connecting each DENSE optimizer step.
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
- [x] Verify the reported update-brake sequence without changing valid transition logic.
- [x] Materialise top Plotly axes with invisible `x2` anchors and restore the final absolute heatmap title.
- [x] Keep the Workspace title in the card header rather than the Plotly chart body.
- [x] Join each DENSE step's crosses with faint straight segments without linking different steps or runs.
- [x] Commit and publish the top-axis repair, then verify dependency-backed CI.

## G. DENSE optimizer-step history refinement

- [x] Give every DENSE optimizer step a stable, random-looking colour shared by its sampled scalars.
- [x] Reduce DENSE cross size and stroke weight while retaining faint per-step connectors.
- [x] Replace scalar-coordinate legend entries with one optimizer-step entry per recorded step.
- [x] Remove `oldest` / `newest` language from DENSE weight titles and legends.
- [x] Add per-weight-chart `Step count` and `Step window` (`Rolling` / `From zero`) display preferences.
- [x] Base all DENSE history filtering on recorded optimizer steps, never probe number.
- [x] Add semantic coverage for step colours, legend ownership, marker styling and both history-window modes.
- [x] Commit and publish this DENSE refinement when requested.

## H. Workspace legend and current-weight display preference

- [x] Remove weight-chart legends from Workspace without changing single-run legends or hover identity.
- [x] Add `Current weights only` to every weight chart's Display Preferences.
- [x] Persist the toggle independently for each run/chart and each Workspace chart.
- [x] In Workspace, retain the newest recorded optimizer step independently for every visible run.
- [x] Run focused browser semantics, source-contract, syntax and regression tests.
- [x] Review and commit the completed tree; publish it when requested.

## I. THOG integer-layer line-segment view

- [x] Verify current-only Workspace filtering cannot retain two optimizer steps from one run.
- [x] Expose exact THOG integer-layer weights in figure metadata without changing training or default charts.
- [x] Add per-weight-chart `Join with line segments` to Display Preferences.
- [x] Render THOG integer weights as open circles joined by straight segments while preserving DENSE crosses.
- [x] Scope the setting independently to each run/chart and each Workspace chart.
- [x] Add focused core, browser-semantic, source-contract and regression coverage.
- [x] Review and commit the completed tree; publish it when requested.

## J. Workspace latest-weight enforcement and Weights group settings

- [x] Reproduce the missing final-render coverage in the Workspace current-only path.
- [x] Enforce newest-recorded-step filtering after the complete Workspace presentation chain.
- [x] Add Run-scoped and Workspace-scoped Weights group settings.
- [x] Make individual weight-chart settings override inherited group settings.
- [x] Add focused browser-semantic, source-contract, syntax and regression coverage.
- [x] Review and commit the completed tree.
- [ ] Publish it when requested.

## K. INSTRA regression repair — selected coupling, segments, chrome and heatmap

- [x] Reproduce the stale THOG fallback after matched-coupling filtering.
- [x] Prevent Runs and Workspace from substituting a different recorded coupling.
- [x] Keep exact THOG integer-segment conversion on the selected trace path.
- [x] Remove redundant Plotly weight titles, right legends and duplicate top-axis titles.
- [x] Key historical coupling metadata by optimizer update and scalar id.
- [x] Keep heatmaps visible whenever capture is enabled or stored heatmap data exist.
- [x] Add Node, Python and real-Firefox coverage for the repaired contract.
- [ ] Publish and require both focused and broad CI success.
