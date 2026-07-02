# THOG2 Stage 1 Mathematical Core Record

**Stage:** 1 — Mathematical core  
**Branch:** `feature/sheet-stage-1-core`  
**Status:** Provisional; acceptance evidence pending CI execution.

## Scope implemented

- normalized depth and fixed left-to-right row coordinates;
- first-kind Chebyshev recurrence;
- deterministic reduced-QR stabilization with positive `R` diagonal;
- basis caching by geometry, basis version, device, and runtime dtype;
- non-persistent basis ownership;
- proportional row-order derivation;
- packed attention, attention output, MLP expansion, MLP contraction, LayerNorm, and optional bias family geometry;
- analytical sheet and dense-equivalent parameter counts;
- the complete S1-01 through S1-15 CPU test inventory;
- high-order calibration through row order 1024.

## Explicit exclusions

Stage 1 does not implement coefficient state, weight materialisation, initialization, SheetGPT, optimizer integration, training integration, activation checkpointing, checkpoint lifecycle, inference, GPU execution, or DDP.

## Acceptance state

This record shall be updated with the exact CI run, calibration results, failures, reruns, and final disposition before merge.
