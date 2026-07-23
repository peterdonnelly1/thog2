# THOG2 Two-Compressor Geometry Interface

## Requirements Specification

**Version:** 0.2  
**Date:** 24 July 2026  
**Status:** Implementation baseline  
**Development branch:** `GEOMETRY_REVISITED`

> Each `--select-element` occurrence names one registered, permitted geometry. The registry—not ad hoc axis accumulation—is authoritative. Phase 1 supports one optional DEPTH compressor and one shared compressor for all selected non-DEPTH elements.

## 1. Executive Summary

THOG2 will expose geometry through a small, explicit registry of semantically permitted elements. A command-line selection does not construct a geometry by combining arbitrary axes. Each occurrence of `--select-element` names one registered geometry whose axes and implied element type are already defined.

The public element types in this phase are **CURVE**, **SHEET** and **SHEET_SET**. A SHEET_SET is a compound registered attention element: a set of independent two-dimensional sheets separated by hard semantic boundaries such as Q/K/V role and attention head. CURVE_SET is not a public type; a CURVE may simply have many independent instances.

DEPTH is selected separately with `--select-depth`. When active, it is universal across repeated Transformer-block elements. Phase 1 permits one compressor for DEPTH and one shared compressor for all selected non-DEPTH elements. The interface remains element-scoped so this restriction can later be relaxed without renaming selectors or redesigning the command line.

**Phase 1 scope:** fully support the registered CURVE, SHEET and SHEET_SET selections while leaving existing legacy BLOCK materialisers untouched. A configuration that combines DEPTH with a complete two-axis SHEET or SHEET_SET implies a three-dimensional field and is rejected by the systematic materialiser in this phase.

## 2. Scope and Governing Decisions

### 2.1 In scope

- A canonical permitted-geometry registry for DEPTH, MLP and attention.
- Repeatable `--select-element` selection, plus the separate universal `--select-depth` control.
- Element- and axis-scoped `--option` assignments.
- Phase 1 validation of one DEPTH compressor and one shared non-DEPTH compressor.
- Full systematic support for registered CURVE, SHEET and SHEET_SET selectors.
- Normal startup reporting of every selected element and its implied type.
- Resolved-plan persistence in checkpoints, run identity and experiment metadata.
- Compatibility with existing short options and legacy presets throughout this phase.

### 2.2 Explicitly out of scope

- Compression across `ATTENTION_HEAD` or `QKV_ROLE`.
- Generic ROW or COLUMN selectors whose meaning changes between elements.
- Systematic BLOCK or BLOCK_SET selectors and materialisers.
- Per-element compressor families during Phase 1.
- Removal or reinterpretation of existing legacy `mlp_block`, `head_aware_block` or `full_block` paths.
- Silent migration of old checkpoint semantics.

### 2.3 Governing decisions

| Decision area | Requirement |
|---|---|
| Selection unit | One `--select-element` occurrence selects one registered permitted geometry. |
| Registry authority | The parser resolves selectors through the registry; it does not manufacture arbitrary axis combinations. |
| Bare element | A bare registered element such as `MLP_UP` or `ATTENTION_QKV` selects its registered complete geometry. |
| Qualified element | A qualified selector such as `MLP_UP.MLP_HIDDEN` selects one registered CURVE. |
| DEPTH | Optional, universal when selected, and controlled separately by `--select-depth`. |
| Public types | CURVE, SHEET and SHEET_SET only. |
| Short options | Existing single-letter meanings remain unchanged throughout this phase. |

### 2.4 Terminology

| Term | Definition |
|---|---|
| Element | A semantically meaningful projection or compound attention geometry, such as `MLP_UP` or `ATTENTION_QKV`. |
| Selector | A canonical registry key supplied to `--select-element`. |
| Axis | A semantically meaningful compressible coordinate, such as `MLP_HIDDEN` or `ATTENTION_HEAD_CHANNEL`. |
| Free/dense axis | Any axis not selected for compression. It remains explicit; this is implicit and need not be restated in every plan. |
| SHEET_SET | A compound registered attention element containing independent sheets separated by hard semantic boundaries. |
| Implied element type | The CURVE, SHEET or SHEET_SET type declared by the selected registry entry. |

## 3. Permitted Geometry Registry

The registry is the permanent semantic authority. It specifies the compressed axes and implied element type for each selector. Physical tensor orientation, packed storage layout and implementation class names do not alter these meanings.

| Selector | Type | Compressed axis/axes | Multiplicity / notes |
|---|---|---|---|
| `DEPTH` | CURVE | DEPTH | Universal across repeated Transformer-block elements |
| `MLP_UP.MLP_HIDDEN` | CURVE | MLP_HIDDEN | One curve per free MLP_D_MODEL coordinate |
| `MLP_UP.MLP_D_MODEL` | CURVE | MLP_D_MODEL | One curve per free MLP_HIDDEN coordinate |
| `MLP_UP` | SHEET | MLP_HIDDEN × MLP_D_MODEL | Complete registered MLP_UP sheet |
| `MLP_DOWN.MLP_HIDDEN` | CURVE | MLP_HIDDEN | Semantic hidden axis despite reversed physical orientation |
| `MLP_DOWN.MLP_D_MODEL` | CURVE | MLP_D_MODEL | Semantic model axis despite reversed physical orientation |
| `MLP_DOWN` | SHEET | MLP_HIDDEN × MLP_D_MODEL | Complete registered MLP_DOWN sheet |
| `ATTENTION_QKV.ATTENTION_D_MODEL` | CURVE | ATTENTION_D_MODEL | Independent instances across QKV_ROLE × HEAD × HEAD_CHANNEL |
| `ATTENTION_QKV.ATTENTION_HEAD_CHANNEL` | CURVE | ATTENTION_HEAD_CHANNEL | Independent instances across QKV_ROLE × HEAD × D_MODEL |
| `ATTENTION_QKV` | SHEET_SET | ATTENTION_D_MODEL × ATTENTION_HEAD_CHANNEL | One sheet per QKV_ROLE × ATTENTION_HEAD |
| `ATTENTION_OUTPUT.ATTENTION_D_MODEL` | CURVE | ATTENTION_D_MODEL | Independent instances across HEAD × HEAD_CHANNEL |
| `ATTENTION_OUTPUT.ATTENTION_HEAD_CHANNEL` | CURVE | ATTENTION_HEAD_CHANNEL | Independent instances across HEAD × D_MODEL |
| `ATTENTION_OUTPUT` | SHEET_SET | ATTENTION_D_MODEL × ATTENTION_HEAD_CHANNEL | One sheet per ATTENTION_HEAD |

### 3.1 Registry semantics

- `MLP_UP` and `MLP_DOWN` are ordinary two-axis sheets.
- `ATTENTION_QKV` and `ATTENTION_OUTPUT` are compound SHEET_SET entries because head and Q/K/V-role boundaries create independent sheets.
- Qualified attention selectors remain CURVEs. Their many independent instances do not create a separate CURVE_SET type.
- A bare selector is not shorthand for “all axes currently known”. It is the stable name of one specific registered geometry whose expansion is persisted in the resolved plan.

### 3.2 Prohibited selectors and crossings

| Rejected target | Reason |
|---|---|
| `ATTENTION_HEAD` | Heads are independently operating units and remain hard semantic boundaries. |
| `QKV_ROLE` | Q, K and V are distinct functions and remain independent sets. |
| Packed QKV rows | A flat physical axis crosses Q/K/V and head boundaries. |
| Flattened attention-output input | A flat HEAD × HEAD_CHANNEL span crosses head boundaries. |
| Generic ROW / COLUMN | The same physical orientation has different semantic meaning across elements. |
| Cross-element geometry | MLP_UP × MLP_DOWN or attention × MLP has no approved semantic basis. |

## 4. Canonical Command-Line Interface

### 4.1 Selection controls

```bash
--select-depth
--select-element SELECTOR
```

`--select-depth` is Boolean and universal. `--select-element` is repeatable. Each occurrence must match exactly one registry entry.

```bash
--select-element MLP_UP.MLP_HIDDEN
--select-element ATTENTION_QKV
```

### 4.2 Option grammar

```bash
--option TARGET.PROPERTY=VALUE
```

| Target kind | Example | Purpose |
|---|---|---|
| DEPTH element | `DEPTH.compressor=chebyshev` | Select DEPTH compressor family. |
| DEPTH axis | `DEPTH.order=32` | Set retained DEPTH order. |
| Element | `MLP_UP.compressor=dct` | Select compressor for the selected geometry of MLP_UP. |
| Axis | `MLP_UP.MLP_HIDDEN.order=256` | Set retained order for one constituent axis. |
| Axis | `MLP_UP.MLP_HIDDEN.group_size=128` | Set a compressor-specific physical option. |

### 4.3 Selection is not axis accumulation

Two repeated qualified selectors do not implicitly create a sheet. `MLP_UP.MLP_HIDDEN` and `MLP_UP.MLP_D_MODEL` are two CURVE registry entries. The complete MLP_UP SHEET is selected directly with:

```bash
--select-element MLP_UP
```

Overlapping selections on the same tensor element are rejected.

### 4.4 Existing short options remain stet

| Short option | Permanent semantic meaning |
|---|---|
| `-P` | DEPTH retained order |
| `-Q` | ATTENTION_D_MODEL retained order |
| `-J` | ATTENTION_QKV.ATTENTION_HEAD_CHANNEL retained order |
| `-O` | ATTENTION_OUTPUT.ATTENTION_HEAD_CHANNEL retained order |
| `-X` | MLP_D_MODEL retained order |
| `-Y` | MLP_HIDDEN retained order |
| `-B`, `-v` and existing preset controls | Remain unchanged throughout this phase; migration is additive, not disruptive. |

## 5. Phase 1 Compressor and Capability Model

### 5.1 Two-compressor restriction

| Scope | Phase 1 rule |
|---|---|
| DEPTH | At most one compressor family/version, applied to the universal DEPTH selection. |
| Non-DEPTH elements | Exactly one shared compressor family/version across all selected non-DEPTH elements. |

The command-line syntax remains element-scoped. A later phase may permit different compressors per element without changing selector names or option grammar. During Phase 1, differing non-DEPTH compressor assignments are a configuration error.

### 5.2 Supported selection combinations

| Selection pattern | Phase 1 status | Reason |
|---|---|---|
| DEPTH only | Supported | Universal one-dimensional compression. |
| One or more non-overlapping CURVE selectors | Supported | One shared non-DEPTH compressor. |
| One or more non-overlapping SHEET / SHEET_SET selectors | Supported | Full systematic SHEET(S) scope. |
| DEPTH + CURVE selector(s) | Supported | Two-compressor path; no complete two-axis intra element is selected. |
| DEPTH + SHEET or SHEET_SET | Rejected in Phase 1 | Would require a systematic three-dimensional field. |
| Overlapping selectors for the same tensor element | Rejected | Would assign multiple geometric representations to the same parameters. |

### 5.3 Compressor capability declarations

Each compressor implementation declares the dimensionalities and physical options it supports. A request may be semantically valid yet unavailable in the current compressor version—for example, a future two-dimensional JPEG-like implementation for `MLP_UP`.

## 6. Resolution, Console Output and Diagnostics

### 6.1 Resolved geometry plan

The plan records schema version, DEPTH state, selected registry keys, implied element types, expanded axes, compressor families and versions, retained orders, axis options, capability result and preset provenance.

### 6.2 Mandatory normal startup output

```text
Selected geometry elements
--------------------------
DEPTH
  implied type:       CURVE
  compressor:         CHEBYSHEV
  order:              32

MLP_UP
  implied type:       SHEET
  compressed axes:    MLP_HIDDEN × MLP_D_MODEL
  compressor:         DCT
  orders:             MLP_HIDDEN=256, MLP_D_MODEL=64

ATTENTION_QKV
  implied type:       SHEET_SET
  compressed axes:    ATTENTION_D_MODEL × ATTENTION_HEAD_CHANNEL
  independent sheets: QKV_ROLE × ATTENTION_HEAD
  compressor:         DCT
```

### 6.3 Explain mode

```bash
--explain-geometry
```

Explain mode resolves presets, selectors, compressor assignments, orders, physical shapes, coefficient shapes and capability status, then exits before model construction or training.

## 7. Validation, Identity and Compatibility

| Stage | Responsibility |
|---|---|
| 1. Syntax | Parse `--select-element` and `--option` grammar. |
| 2. Registry | Resolve every selector to one permitted registry entry. |
| 3. Semantic consistency | Reject overlaps, hard-boundary crossings and invalid option targets. |
| 4. Phase capability | Enforce one shared non-DEPTH compressor and reject DEPTH + complete SHEET(S). |
| 5. Compressor capability | Check dimensionality, versions and compressor-specific options. |
| 6. Runtime shape | Check physical lengths, retained orders and divisibility constraints. |

Identity is based on the fully resolved plan: explicit selector keys, implied types, expanded axes, compressors and versions, orders, physical options and materialisation versions. A preset label alone is insufficient. Old checkpoints either map unambiguously or fail clearly; silent reinterpretation is prohibited.

## 8. Low-Risk Implementation Approach

### 8.1 Stage A — semantic foundation and UI

- Add the permitted-geometry registry and immutable selector metadata.
- Add `ResolvedGeometryPlan` and make both presets and new arguments resolve into it.
- Add `--select-depth`, repeatable `--select-element`, `--option` and `--explain-geometry`.
- Add console reporting, validation and identity persistence.
- Map currently supported selections to existing materialisers without changing numerical behaviour.

### 8.2 Stage B — complete systematic SHEET(S)

- Generalise element metadata so physical MLP widths are derived rather than hard-coded to `4 × d_model`.
- Support DEPTH-off non-DEPTH compression, including isolated MLP_HIDDEN experiments.
- Implement every registered MLP SHEET and attention SHEET_SET entry through one restrained separable geometry layer.
- Preserve QKV_ROLE and ATTENTION_HEAD boundaries through explicit tensor decomposition and sentinel tests.
- Retain a direct path to existing optimised materialisers until the systematic path demonstrates parity.

### 8.3 Stage C — adoption and optimisation

- Benchmark numerical equivalence, memory and wall time against current DEPTH, JPEG-like and legacy geometry paths.
- Add direct factorised execution only after correctness and checkpoint behaviour are stable.
- Migrate wrappers gradually; keep short options and legacy presets working throughout the phase.
- Do not modify or remove legacy BLOCK materialisers as part of this work.

**Risk-control rule:** the new systematic path is additive. Existing `BLOCK_enhancements_group` behaviour remains reproducible; `GEOMETRY_REVISITED` becomes the sole development branch for this work. Numerical refactoring begins only after the registry, resolver and diagnostics are covered by tests.

## 9. Requirements and Acceptance Criteria

| ID | Requirement |
|---|---|
| UI-001 | The CLI shall provide `--select-depth` and repeatable `--select-element SELECTOR` controls. |
| UI-002 | Each `--select-element` occurrence shall resolve to exactly one registered permitted geometry. |
| UI-003 | The CLI shall provide `--option TARGET.PROPERTY=VALUE` for element- and axis-scoped settings. |
| UI-004 | Existing single-letter options and their meanings shall remain unchanged throughout this phase. |
| REG-001 | The registry shall contain the complete selector set defined in Section 3. |
| REG-002 | Bare MLP selectors shall imply SHEET; bare attention selectors shall imply SHEET_SET. |
| REG-003 | CURVE_SET, BLOCK and BLOCK_SET shall not be public systematic types in Phase 1. |
| SEM-001 | ATTENTION_HEAD and QKV_ROLE shall not be selectable compression axes. |
| SEM-002 | Overlapping selectors for the same tensor element shall be rejected. |
| CMP-001 | Phase 1 shall support one optional DEPTH compressor and one shared non-DEPTH compressor. |
| CMP-002 | Differing non-DEPTH compressor assignments shall be rejected with a specific Phase 1 diagnostic. |
| CAP-001 | Every compressor implementation shall declare supported dimensionalities and physical constraints. |
| MAT-001 | All registered MLP SHEET and attention SHEET_SET selectors shall be implemented in the systematic path. |
| MAT-002 | The systematic path shall support non-DEPTH compression with DEPTH disabled. |
| MAT-003 | MLP axis lengths shall be derived from element metadata and shall not assume a 4× projection factor. |
| DIA-001 | Normal startup output shall display every selected selector and its implied element type. |
| DIA-002 | `--explain-geometry` shall resolve and validate geometry without starting training. |
| ID-001 | The fully resolved geometry plan shall be persisted in checkpoint and run identity. |
| LEG-001 | Legacy BLOCK materialisers and presets shall remain untouched by the systematic SHEET(S) implementation. |

## Appendix A — Complete Examples

### A.1 CURVE — MLP hidden only

```bash
--select-element MLP_UP.MLP_HIDDEN \
--option MLP_UP.compressor=jpeg_like \
--option MLP_UP.MLP_HIDDEN.order=8 \
--option MLP_UP.MLP_HIDDEN.group_size=128
```

### A.2 SHEET — complete MLP_UP

```bash
--select-element MLP_UP \
--option MLP_UP.compressor=dct \
--option MLP_UP.MLP_HIDDEN.order=256 \
--option MLP_UP.MLP_D_MODEL.order=64
```

### A.3 SHEET_SET — complete QKV attention element

```bash
--select-element ATTENTION_QKV \
--option ATTENTION_QKV.compressor=dct \
--option ATTENTION_QKV.ATTENTION_D_MODEL.order=256 \
--option ATTENTION_QKV.ATTENTION_HEAD_CHANNEL.order=6
```

### A.4 DEPTH plus a non-DEPTH CURVE

```bash
--select-depth \
--select-element MLP_UP.MLP_HIDDEN \
--option DEPTH.compressor=chebyshev \
--option DEPTH.order=32 \
--option MLP_UP.compressor=jpeg_like \
--option MLP_UP.MLP_HIDDEN.order=8 \
--option MLP_UP.MLP_HIDDEN.group_size=128
```

### A.5 Rejected Phase 1 three-dimensional combination

```bash
--select-depth \
--select-element ATTENTION_QKV \
--option DEPTH.compressor=chebyshev \
--option ATTENTION_QKV.compressor=dct
```

### A.6 Rejected Phase 1 compressor mismatch

```bash
--select-element MLP_UP \
--select-element ATTENTION_QKV \
--option MLP_UP.compressor=dct \
--option ATTENTION_QKV.compressor=haar
```

### A.7 Rejected overlap

```bash
--select-element MLP_UP \
--select-element MLP_UP.MLP_HIDDEN
```
