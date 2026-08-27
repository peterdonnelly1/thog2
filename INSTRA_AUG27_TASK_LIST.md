# INSTRA August 27 Task List

Branch: `depth_weight_curves_and_observational_probes`

## Required changes

- [x] 1. Add a Weights-group `gradient` button after `latest step`, separated by twice the normal control gap. When multiple curves are visible, colour by step from lightest (earliest) through the run colour at the middle step to darkest (latest).
- [x] 2a. Replace `CURVES` with capture-start/end columns `C_s` and `C_e`.
- [x] 2b. Replace `PROBES` with capture-start/end columns `P_s` and `P_e`.
- [x] 3.1. Keep individual weight-chart controls accessible in the Weights group title bar while a weight chart is maximized.
- [x] 3.2. In Workspace, inherit each run's Runs line-segment preference until the user explicitly overrides the Workspace preference.
- [x] 4.1. Repair Workspace retained-step intersection and the false no-overlap error.
- [x] 4.2. Do not show `No retained weight steps are available.` while curves are present/loading.
- [x] 4.3. Clear Workspace range errors on return to Runs.
- [x] 4.4. Prevent stale `showing steps ...` copy from preceding the correct range.
- [x] 5.1. Show only `capture window A-B`; remove redundant `data available A-B` copy.
- [x] 5.2. Replace `show weights for steps` with `steps`.
- [x] 6.1-6.3. Use `Loading weight curves...` while retained data is pending; reserve no-data copy for authoritative completed-empty responses.
- [x] 7. Determine and repair why the first curve can appear many logging intervals after capture begins.
- [x] 8. Determine whether `history_length` is redundant; remove or redefine it only if retention can remain bounded and capture semantics stay explicit.

## Acceptance gates

- [x] JavaScript syntax and deterministic dashboard regressions.
- [x] Python storage/server/lifecycle regressions.
- [x] Live SQLite + HTTP acceptance for capture ranges and loading states.
- [ ] Real Firefox Runs/Workspace/fullscreen acceptance.
- [ ] Broad head-versus-base CPU comparison with zero new regressions.
- [ ] Publish exact tested tree and provide download stanza.

## Binding invariants

- A completed capture window stops adding snapshots; it never hides or discards retained snapshots.
- Workspace overlap is the intersection of retained ranges for the currently visible selected runs.
- Groups open only through an explicit user action.
- Existing run data must remain readable.
