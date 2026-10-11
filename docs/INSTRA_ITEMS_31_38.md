# Instra Recipe and PREMAT fixes: items 31-38

Branch: `cpu_materialisation`. Implementation base: `439d724864dd5ed48027f148c0b7d670ebfd1811`.

| Item | Behaviour |
| --- | --- |
| 31 | Hover help lists the canonical enum choices before explanatory text. CPU numeric controls and layer batches show their valid ranges. Matrix selector help identifies 1=QKV, 2=O, 3=UP and 4=DOWN. |
| 32 | Every enum and boolean field has an explicit native Choices popup beside its editable input. Open-ended controls use preset suggestions without restricting arbitrary valid values. Comma-separated sweeps remain editable. Inactive and automatically controlled fields disable their picker. |
| 33 | `--premat` is the first field in CPU Premat. |
| 34 | Progress State cells and their legend use the same CSS classes and the existing legend palette. Other Runner tabs retain their existing state colours. |
| 35 | `--premat_instra_playback complete_step` is the default. Its chart button switches between Complete optimiser step and Latest data. Skip interrupts the current animation, clears queued older captures and starts the newest received complete microstep immediately. |
| 36 | Successful new, unchanged, copied and in-place Recipe saves return to Frequently Used. |
| 37 | Recapitulation, inspector and history labels say Microstep N of Optimiser Step M. |
| 38 | Worker metadata cannot replace scheduler event identity, timestamp, sequence, state or outcome. Retained CPU events with an intact `event_type` are interpreted correctly without changing their raw evidence. |

## Playback semantics

Complete optimiser step preserves every frame and queued sibling microstep of the active optimiser step. New data does not cancel its timer. Pause preserves its position; Resume continues it. Latest data immediately selects the newest complete captured microstep, including after resuming a pause. Both modes process incoming batches once rather than repeatedly rendering every snapshot in a Latest batch.

The mode button overrides the recorded run preference until selecting another run. Skip resumes playback and jumps immediately but leaves the selected mode unchanged. When no newer data is queued, Skip abandons the current animation and uses the newest microstep in the next incoming batch.

To bound a long backlog, whole future optimiser steps may be discarded after the pending queue exceeds 128 microsteps. Siblings of the active step and the newest queued step are preserved; a large accumulation count is not truncated at 128. Detailed retained history remains independently available. Playback cannot reconstruct microsteps that were never captured or were already removed from the backend's retained history before the dashboard received them.

Capture remains independent of playback. Use `--premat_instra_microsteps all` to capture every original-forward microstep of sampled updates, and `--log-interval 1` to sample every optimiser step. First-only capture still has only one microstep to replay.

## Apply and smoke-test

Pull the branch and restart Instra; refresh the browser. Training processes already running retain their original Python code. Start a new run to use the corrected scheduler reporting; existing retained CPU events can still be recovered when their authoritative `event_type` survives.

1. In Recipes, open CPU Premat. Confirm that `--premat` leads the fields, enable CPU PREMAT, and select preparation, transfer timing and layer-batch presets through Choices. Enter an arbitrary valid batch such as 17; it must remain accepted. Save from another category and confirm Frequently Used is selected.
2. Open Progress for a grid containing several run states. Each State cell must match its legend colour.
3. In PREMAT, select Complete optimiser step and slow the state-duration slider. New captures must not interrupt the current animation or its sibling microsteps. Click Skip; it must jump to the newest received microstep. Switch to Latest data and confirm incoming data interrupts immediately.
4. Inspect/export a new six-microstep CPU run. Check that matrix outcomes agree with the raw `matrix_acquired` events and that completed matrices are no longer left as NOT YET REACHED because of worker metadata. A fallback reason does not establish a hit, and these reporting changes do not establish a throughput benefit.

## Regression checks

`tests/test_cpu_materialisation_reporting.py` exercises real scheduler transitions with mocked CUDA transfers and deliberately conflicting worker metadata, plus CLI/Runner/config propagation. `tests/instra_premat_playback_regression.js` covers six-microstep playback, 180-microstep queue pressure, pause, Skip, Latest mode, run resets and 384 explicit CPU outcomes in both intact and historically overwritten events. `tests/instra_recipes_playback_browser.js` checks native popups, sweeps, all four Save forms, all eleven Progress states and chart controls in Chromium and Firefox.

The CPU materialisation workflow runs the new Python tests automatically and explicitly runs the new browser checks. Validation results are recorded in `evidence/instra_items_31_38_validation.json`.

Local validation: 2,231 ordinary Python test cases passed in the broad candidate run. Its 66 failing testcase identities and all failure counts match the original branch exactly. The focused run passed 600 cases with one inherited chart-route failure; the independent stage/resume comparison also has identical candidate and baseline failures. All 42 JavaScript suites passed. Chromium and Firefox passed the new Recipe/playback checks, microstep/Nsight checks, legacy settings checks and 60-second CPU-evidence soak checks, with no page errors or growing DOM/trace counts. This environment has CPU-only PyTorch, so the scheduler regression uses mocked CUDA readiness and no live GPU performance claim is made.
