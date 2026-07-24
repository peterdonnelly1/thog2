# THOG2 Geometry Revisited — Phase 1 Test Plan

**Branch:** `GEOMETRY_REVISITED`  
**Date:** 24 July 2026

## 1. Registry tests

Verify:

- the exact twelve permitted non-DEPTH selectors;
- MLP bare selectors imply SHEET;
- attention bare selectors imply SHEET_SET;
- qualified attention selectors remain CURVE, with independent instance metadata;
- no public CURVE_SET, BLOCK, or BLOCK_SET type;
- hard role/head boundaries appear only as independent indices.

## 2. Selection semantics

Verify:

- repeatable selectors remain separate registered geometries;
- two qualified MLP axes remain two CURVES rather than an implicit SHEET;
- duplicate selectors fail;
- complete and qualified selectors for the same element fail as overlaps;
- DEPTH plus complete SHEET/SHEET_SET fails as a three-dimensional field.

## 3. Option grammar and validation

Verify:

- `TARGET.PROPERTY=VALUE` parsing;
- qualified axis targets are split correctly;
- inactive targets fail;
- order and group size require positive integers;
- `family@version` and explicit compressor version handling;
- group size is rejected for compressors that do not support it;
- different non-DEPTH compressor assignments produce the Phase 1 diagnostic.

## 4. Capability and adapter tests

Verify:

- DEPTH only maps to the existing `depth` preset;
- DEPTH plus `MLP_UP.MLP_HIDDEN` with JPEG-like maps to `jpeg_like_v1`;
- selected order and group size reach the adapted legacy configuration;
- other registered geometries resolve successfully for explain mode but fail training before model allocation with a Phase 2 message.

## 5. CLI tests

Verify:

- repeatable `--select-element` and `--option` parsing;
- systematic selections imply `sheet` model type when omitted;
- non-default legacy geometry controls cannot be mixed with systematic selectors;
- explain mode accesses neither dataset nor model;
- normal startup reporting includes the implied element type.

## 6. Identity tests

Verify:

- plan dictionary validates against schema and registry versions;
- plan survives `OwtRunConfig` → `TrainingConfig` conversion;
- plan appears in canonical and compact identity metadata;
- compatibility signatures compare the plan;
- old configurations with no plan remain valid.

## 7. Regression checks

Run:

```bash
python -m compileall -q sheet tests run_thog2_owt.py
bash -n current_scruffy_train_OWT.sh
bash -n current_dreedle_train_OWT.sh
python -m unittest tests.test_geometry_registry_phase1 tests.test_geometry_phase1_cli
python -m unittest discover tests
```
