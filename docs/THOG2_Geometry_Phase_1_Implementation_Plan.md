# THOG2 Geometry Revisited — Phase 1 Implementation Plan

**Branch:** `GEOMETRY_REVISITED`  
**Date:** 24 July 2026

## 1. Objective

Implement the permanent systematic geometry UI and semantic registry while preserving all existing numerical paths. Phase 1 ends with a stable resolved-plan and materialiser-adapter boundary. It does not implement the remaining Phase 2 geometries.

## 2. Staging

### Stage 1 — Registry and plan model

Add `sheet/geometry_registry.py` containing:

- the authoritative twelve-entry selector registry;
- public types `CURVE`, `SHEET`, and `SHEET_SET`;
- compressor capability metadata;
- option parsing and canonicalisation;
- overlap, boundary, dimensionality, and Phase 1 compressor validation;
- immutable resolved selection and plan records;
- checkpoint-plan validation;
- console formatting;
- current-materialiser adapters.

### Stage 2 — CLI integration

Extend `run_thog2_owt.py` with:

- `--select-depth`;
- repeatable `--select-element`;
- repeatable `--option`;
- `--explain-geometry`;
- systematic plan resolution before model allocation;
- automatic geometry summary before training;
- adapter conversion to existing `depth` and `jpeg_like_v1` configurations.

Legacy CLI behavior remains the default when no systematic selector is present.

### Stage 3 — Identity persistence

Extend `OwtRunConfig` and `TrainingConfig` with an optional resolved geometry plan. Include it in:

- canonical configuration;
- compact identity metadata;
- checkpoint compatibility signature;
- telemetry configuration.

Old configurations receive `None` and remain compatible.

### Stage 4 — Wrapper exposure

Allow both current OWT wrappers to accept the systematic long options before their existing `--` separator. Preserve every established short option and grid behavior. Explain mode shall bypass artifact-name resolution and training.

### Stage 5 — Verification

Run:

- Python compile checks;
- shell syntax checks for both wrappers;
- focused Phase 1 registry and CLI tests;
- repository-wide CPU unit tests.

## 3. Low-risk constraints

- Do not rewrite existing trajectories.
- Do not alter legacy block presets.
- Do not change existing defaults when systematic selectors are absent.
- Do not reinterpret qualified selectors as accumulated axes.
- Do not permit systematic BLOCK/BLOCK_SET.
- Fail before dataset/model construction for unimplemented Phase 2 geometries.
