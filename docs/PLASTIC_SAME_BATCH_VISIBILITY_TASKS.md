# PLASTIC same-batch operator visibility tasks

Parent implementation: `docs/PLASTIC_SAME_BATCH_ALL_PROBES_IMPLEMENTATION_LOG.md`
Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`
PR: #33
Started: 2026-08-09
Validated code head: `5e6e62045d77c53ac4a5f31616e59034f29a9c36`

## Status

IMPLEMENTED; CPU/DDP/DISABLED-EQUIVALENCE VALIDATED. CORRECTED REAL-CUDA SMOKE REMAINS EXTERNAL.

## Task list

- [x] Diagnose user report that enabled same-batch mode was not visible in startup output.
- [x] Add explicit compact startup visibility: `plastic fine: ... same_batch=true|false`.
- [x] Fix detailed startup visibility for direct `__main__` execution and show `plastic__layer_count__same_batch_all_probes: true|false`.
- [x] Regression-check that every parser-exposed `plastic__*` control is represented by the detailed PLASTIC header surface.
- [x] Make operator-facing P numbering window/batch-local: every fresh batch starts at P1.
- [x] Render probe labels compactly as `P1`, `P2`, ... with no padding between P and the number.
- [x] Start every probe section at terminal-visible column 360; the probe tail wins over an overlong sampled vector.
- [x] Suppress the directional decision summary/provenance until the configured evidence window is complete.
- [x] Use the full-size `constants.py` triangle glyphs `▼` / `▲` for directional display.
- [x] Highlight only a committed `▼` / `▲` outcome in `constants.BOLD + constants.YELLOW`.
- [x] Highlight a changed existing sampled-layer value with `constants.YELLOW` for exactly its first changed row; unchanged repeats are unhighlighted.
- [x] Restore a fixed seven-character numeric field after `g nrm=`.
- [x] Hide `same_batch W<window>:<ordinal>/<size> B=<digest8>` unless `DEBUG > 9`.
- [x] Retain monotonically increasing global probe sequence/provenance only as internal audit fields for durability/debugging.
- [x] Preserve false/default runtime behavior.
- [x] Final regression run `31294179340`: SUCCESS; broad CPU suite/classifier green.
- [x] Final disabled-equivalence run `31294179362`: SUCCESS.
- [x] Final same-batch-enabled DDP run `31294179291`: SUCCESS.
- [ ] Rerun the corrected real-CUDA same-batch smoke externally before relying on allocator/OOM behavior as fully validated.

## Expected normal operator sequence for W=2

```text
... sampled = [...]                                                P1  probe_Δloss [...]  score_z [...]
... sampled = [...]                                                P2  probe_Δloss [...]  score_z [...]  ▼|▲|? =[...]/2=>● (P1,2)

... sampled = [...]                                                P1  probe_Δloss [...]  score_z [...]
```

`P1`/`P2` begins at terminal column 360. The first probe has no decision summary because it is not qualified to decide. The next fresh evidence batch resets visible numbering to P1. The same-batch forensic marker is hidden at normal `DEBUG=9` and appears only for `DEBUG > 9`.

## Relevant commits

- `b99186fad0ae5633b89bd3e96bf523d8f0ac4a9f` — initial visibility/local-provenance overlay.
- `11f4d8833502235f865e8e987bd984219eb06caf` — compact header shows same-batch mode.
- `98b09ae35171463c37d3e45b093b23c57e9bdfef` — implement requested decision/glyph/colour/P/sample/gradient/debug/header UI semantics.
- `aa3643c8cf8bf0e86e5755dddac7f4dfc7d29b83` — focused UI regression coverage.
- `3351ffda5798babf1e1b04e60f04da1f06c911b3` — make column 360 terminal-tab aware.
- `5e6e62045d77c53ac4a5f31616e59034f29a9c36` — assert terminal-visible column 360; final validated code head.
