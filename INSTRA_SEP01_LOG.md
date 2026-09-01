# INSTRA September 1 Enhancements - Work Log

## Context and plan

2026-09-01: Recovered sandpit at bbb13f0 on initialisation_baselining. GitHub connector confirms published prior Instra log and branch. Read prior task/log files; inspecting baseline snapshot v0.2 spec and current code. An existing modification to docs/PLASTIC_V056_PROGRESS.md is unrelated and must not be staged or overwritten.

Implementation plan: modify existing authoritative rendering hooks, keep z cycling stable across rerenders/magnification, darken gradient endpoint, rename public CLI and metadata with legacy aliases, persist snapshot source config for Overview without loading tensors in the dashboard. Test focused behaviours plus existing JS suite and feasible Python checks. Publish through connector with a fast-forward ref update.

## Implementation and verification

- Added a session-stable circular Workspace drawing order, shared across six weight charts. The small z button uses the existing header that remains available when magnified. Per-run step order, figure sources and run-list order are preserved.
- Gradient endpoint now blends the run colour 30% toward black. Multi-step shading remains ordered by step; single-step rendering is unchanged. Pure black remains black because no darker colour exists.
- Renamed the public CLI destinations, help, console names and prospective stored configuration. Hidden aliases accept both old names and their underscore/hyphen wrapper forms. Runtime environment names and internal materialisation functions remain compatible.
- Overview Summary reports the consumed snapshot basename or '-'. A collapsible/searchable snapshot section follows Config and retains its search/scroll/disclosure state. Artifact Outputs is no longer mounted or rendered.
- Snapshot v0.2 explicitly excludes full source configuration from its tensor payload. A best-effort .overview.json companion stores full source training hyperparameters without altering the snapshot format. Import checks both physical and tensor hashes, and persists details in run metadata. Old snapshots still expose physical metadata on new loads. Existing runs can recover source hyperparameters from an exact matching source run in the dashboard catalog; otherwise the missing history is labelled explicitly.
- Passed all 16 tests/*regression.js programs, including the new z-cycle/snapshot Overview behaviour test. Updated superseded assertions for Artifact Outputs, gradient endpoint and CLI names.
- Passed three dependency-light Python unittest cases (including six CLI spelling combinations with true/false publication), all dashboard JavaScript syntax, and Python compilation.
- Browser binaries, PyTorch and pytest are absent in both available Python runtimes. No full-browser or GPU execution is claimed. Existing Firefox gradient acceptance expectations have been updated for execution in the normal project environment.
- Unrelated docs/PLASTIC_V056_PROGRESS.md modification remains untouched and will be excluded from publication.

## Delivery

Reviewed parent: bbb13f05c24feee456078977126eaac317321c05. Publication uses GitHub connector blobs/tree/commit and a non-forced fast-forward update of initialisation_baselining. The delivery commit contains this log; the final response reports its verified SHA. Download: fetch/switch/pull --ff-only initialisation_baselining, then restart Instra.
