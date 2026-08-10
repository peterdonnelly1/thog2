# PLASTIC same-batch operator visibility implementation log

Parent implementation: `docs/PLASTIC_SAME_BATCH_ALL_PROBES_IMPLEMENTATION_LOG.md`
Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`
PR: #33
Date: 2026-08-09
Validated code head: `5e6e62045d77c53ac4a5f31616e59034f29a9c36`

## Trigger

A real scruffy run showed that the same-batch state machine was working, but the operator console still exposed too much interim decision detail, used visually small arrows, did not make changed sampled coordinates conspicuous enough, allowed probe tails to drift horizontally with sampled-vector length, and omitted the new same-batch control from the detailed PLASTIC header when the runner executed as `__main__`.

## Final governing display semantics

1. Compact startup continues to report `same_batch=true|false`.
2. Detailed startup reports `plastic__layer_count__same_batch_all_probes: true|false`; direct `__main__` execution is explicitly supported.
3. The detailed PLASTIC header is regression-checked against every parser-exposed `plastic__*` control so a new public PLASTIC hyperparameter cannot silently disappear from the header.
4. Visible probe numbering is evidence-window local: `P1`, `P2`, ...; a fresh batch resets to `P1`.
5. Probe labels contain no padding between P and the number.
6. The probe section starts at terminal-visible column 360. Terminal tab expansion is accounted for. If the sampled vector would extend beyond that point, its visible tail is truncated and the probe section wins.
7. The directional decision summary/provenance is not displayed before the configured evidence window is complete. Probe loss and `score_z` remain visible on interim probes.
8. Direction glyphs use `constants.DOWN_ARROW` / `constants.UP_ARROW` (`▼` / `▲`) rather than the smaller arrow glyphs.
9. Only a committed directional outcome (`▼` or `▲`) is emphasized, using `constants.BOLD + constants.YELLOW` and then `constants.R`.
10. Existing sampled-layer values are compared by displayed one-decimal value. A changed existing coordinate is `constants.YELLOW` for the first row showing its new value and plain thereafter while unchanged. Newly appended values are not treated as a change to an existing value.
11. `g nrm=` again owns a fixed seven-character numeric field.
12. The forensic `same_batch W<window>:<ordinal>/<size> B=<digest8>` marker is hidden for routine runs and emitted only when `constants.DEBUG > 9`.
13. Global monotonically increasing probe sequence/provenance remains internal audit state; operator P numbering stays window local.

## Implementation

### `sheet/plastic_depth_same_batch_visibility_patch.py`

This remains the final additive console overlay. The follow-on cleanup in `98b09ae35171463c37d3e45b093b23c57e9bdfef` added:

- full-window readiness metadata for console-only decision-summary gating;
- exact one-row sampled-coordinate change highlighting using `constants.YELLOW`;
- full-size direction glyph normalization;
- explicit bold/yellow committed-direction outcome normalization;
- fixed-width `g nrm` normalization;
- DEBUG-gated same-batch forensic identity;
- compact `P1`/`P2` formatting;
- absolute probe-tail alignment;
- direct-`__main__` startup-module resolution so the detailed header receives the same-batch control during normal script execution.

The first column-360 implementation counted raw tab characters rather than terminal tab stops. That would have aligned the string but not the actual terminal. Commit `3351ffda5798babf1e1b04e60f04da1f06c911b3` corrected visible-width and truncation calculations to use 8-column terminal tab semantics.

### `tests/test_plastic_depth_same_batch_visibility.py`

The final focused tests assert:

- incomplete evidence windows do not show `▼|▲|?` or provenance;
- complete windows do show the decision summary;
- committed `▲` is wrapped exactly in `constants.BOLD + constants.YELLOW`;
- P numbering resets with each fresh batch and renders as `P1`, `P2`, ...;
- the P section begins at 1-based terminal column 360 after ANSI removal and tab expansion;
- the same-batch forensic marker is absent at normal DEBUG and present above 9;
- an existing sampled coordinate changing `19.0 -> 18.9` is yellow exactly once;
- `g nrm=  5.921` and `g nrm= 10.746` retain one common field width;
- startup resolution works through `__main__`;
- every parser-exposed `plastic__*` control is represented on the detailed PLASTIC header surface;
- the public wrapper still reports compact `same_batch=true`.

Focused test commit: `aa3643c8cf8bf0e86e5755dddac7f4dfc7d29b83`. Terminal-column assertion correction: `5e6e62045d77c53ac4a5f31616e59034f29a9c36`.

## Validation

Validated code head: `5e6e62045d77c53ac4a5f31616e59034f29a9c36`.

- `Validate PLASTIC COARSE FINE regression` run `31294179340`: SUCCESS. Production compile and shell syntax passed; broad CPU suite and inherited-failure classifier passed.
- `Validate PLASTIC disabled equivalence` run `31294179362`: SUCCESS.
- `Validate PLASTIC COARSE FINE DDP` run `31294179291`: SUCCESS.

No decision algorithm, score calculation, evidence history, checkpoint state, batch acquisition, or DDP semantics were changed by this UI follow-on.

## External GPU gate

The corrected same-batch CUDA smoke remains useful for allocator/OOM coverage:

```bash
cd ~/git/thog2
git fetch origin PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git checkout PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git reset --hard origin/PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
bash tools/run_plastic_coarse_fine_gpu_smoke.sh
```
