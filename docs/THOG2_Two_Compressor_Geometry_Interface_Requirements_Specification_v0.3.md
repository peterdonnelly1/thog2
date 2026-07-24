# THOG2 Two-Compressor Geometry Interface

## Requirements Specification

**Version:** 0.3  
**Date:** 24 July 2026  
**Status:** Phase 1 implementation baseline  
**Development branch:** `GEOMETRY_REVISITED`

> Each `--select-element` occurrence names one registered, permitted geometry. The registry—not ad hoc axis accumulation—is authoritative. Phase 1 implements the complete UI, registry, validation, resolved-plan identity and adapter boundary. Phase 2 implements the remaining numerical materialisers represented by the registry.

## 1. Purpose and phase boundary

Phase 1 establishes the permanent systematic geometry interface without replacing proven numerical paths. It shall:

- implement `--select-depth`, repeatable `--select-element`, repeatable `--option`, and `--explain-geometry`;
- implement the complete permitted-geometry registry;
- infer and report `CURVE`, `SHEET`, or `SHEET_SET` for every selection;
- enforce the Phase 1 restriction of one optional DEPTH compressor and one shared non-DEPTH compressor;
- persist the fully resolved plan in run and checkpoint identity;
- adapt configurations already supported by existing materialisers;
- reject registered but not-yet-materialised configurations before model allocation with a precise Phase 2 diagnostic;
- leave all existing short options and legacy presets unchanged.

Phase 2 implements every remaining registered CURVE, SHEET, and SHEET_SET numerical path.

## 2. Governing semantic rules

1. `DEPTH` is universal when selected and is selected only with `--select-depth`.
2. Each `--select-element` occurrence selects exactly one registered geometry.
3. Qualified selectors do not accumulate into an implicit higher-dimensional geometry.
4. Bare MLP elements name complete MLP SHEET geometries.
5. Bare attention elements name compound SHEET_SET geometries.
6. `ATTENTION_HEAD` and `QKV_ROLE` are hard semantic boundaries and are never compressed axes.
7. `CURVE_SET` is not a public type. A CURVE may have many independent instances.
8. Systematic `BLOCK` and `BLOCK_SET` are excluded. DEPTH combined with a complete SHEET or SHEET_SET is rejected.
9. Existing legacy block materialisers remain available and untouched.

## 3. Authoritative permitted-geometry registry

| Selector | Compressed axis/axes | Implied type without DEPTH | Implied type with DEPTH | Independent indices |
|---|---|---|---|---|
| `MLP_UP.MLP_HIDDEN` | `MLP_HIDDEN` | `CURVE` | `SHEET` | `MLP_D_MODEL` |
| `MLP_UP.MLP_D_MODEL` | `MLP_D_MODEL` | `CURVE` | `SHEET` | `MLP_HIDDEN` |
| `MLP_UP` | `MLP_HIDDEN × MLP_D_MODEL` | `SHEET` | Rejected | none |
| `MLP_DOWN.MLP_HIDDEN` | `MLP_HIDDEN` | `CURVE` | `SHEET` | `MLP_D_MODEL` |
| `MLP_DOWN.MLP_D_MODEL` | `MLP_D_MODEL` | `CURVE` | `SHEET` | `MLP_HIDDEN` |
| `MLP_DOWN` | `MLP_HIDDEN × MLP_D_MODEL` | `SHEET` | Rejected | none |
| `ATTENTION_QKV.ATTENTION_D_MODEL` | `ATTENTION_D_MODEL` | `CURVE` | `SHEET_SET` | `QKV_ROLE × ATTENTION_HEAD × ATTENTION_HEAD_CHANNEL` |
| `ATTENTION_QKV.ATTENTION_HEAD_CHANNEL` | `ATTENTION_HEAD_CHANNEL` | `CURVE` | `SHEET_SET` | `QKV_ROLE × ATTENTION_HEAD × ATTENTION_D_MODEL` |
| `ATTENTION_QKV` | `ATTENTION_D_MODEL × ATTENTION_HEAD_CHANNEL` | `SHEET_SET` | Rejected | `QKV_ROLE × ATTENTION_HEAD` |
| `ATTENTION_OUTPUT.ATTENTION_D_MODEL` | `ATTENTION_D_MODEL` | `CURVE` | `SHEET_SET` | `ATTENTION_HEAD × ATTENTION_HEAD_CHANNEL` |
| `ATTENTION_OUTPUT.ATTENTION_HEAD_CHANNEL` | `ATTENTION_HEAD_CHANNEL` | `CURVE` | `SHEET_SET` | `ATTENTION_HEAD × ATTENTION_D_MODEL` |
| `ATTENTION_OUTPUT` | `ATTENTION_D_MODEL × ATTENTION_HEAD_CHANNEL` | `SHEET_SET` | Rejected | `ATTENTION_HEAD` |

`DEPTH` itself is a universal `CURVE` over the repeated Transformer-block layer axis.

## 4. Command-line interface

### 4.1 Selection

```bash
--select-depth
--select-element MLP_UP.MLP_HIDDEN
--select-element ATTENTION_QKV
```

`--select-element` is repeatable. Repeating it selects multiple registered geometries. It does not combine qualified selectors into a new geometry.

Thus:

```bash
--select-element MLP_UP.MLP_HIDDEN
--select-element MLP_UP.MLP_D_MODEL
```

selects two CURVES. The registered MLP SHEET is selected with:

```bash
--select-element MLP_UP
```

### 4.2 Element- and axis-scoped options

```bash
--option TARGET.PROPERTY=VALUE
```

Examples:

```bash
--option DEPTH.compressor=chebyshev
--option DEPTH.order=32
--option MLP_UP.compressor=jpeg_like
--option MLP_UP.MLP_HIDDEN.order=8
--option MLP_UP.MLP_HIDDEN.group_size=128
```

Compressor values may include an inline version as `family@version`. `TARGET.compressor_version=version` is also accepted.

### 4.3 Existing short options

All existing single-letter options retain their existing meanings throughout Phase 1. They are not removed, renamed, or repurposed. The resolved systematic plan may use the existing order values as defaults when an explicit axis-scoped `order` is absent.

## 5. Phase 1 compressor rule

The public syntax is element-scoped, but Phase 1 enforces:

- at most one DEPTH compressor family/version;
- exactly one shared non-DEPTH compressor family/version across all selected non-DEPTH elements.

Different non-DEPTH assignments are a specific Phase 1 configuration error. The resolved schema remains capable of storing per-element assignments so the restriction can later be relaxed without redesigning the UI.

## 6. Validation layers

Validation shall occur in this order:

1. Parse `TARGET.PROPERTY=VALUE` syntax.
2. Resolve each selector against the authoritative registry.
3. Reject duplicate and overlapping selectors.
4. Reject options targeting unselected elements or inactive axes.
5. Reject DEPTH plus a complete SHEET or SHEET_SET.
6. Enforce the shared non-DEPTH compressor rule.
7. Validate compressor dimensionality and physical options.
8. Resolve current materialiser capability.
9. Reject registered but unimplemented Phase 2 geometries before model allocation.

A registered but unavailable configuration is not described as invalid. The diagnostic shall state that it is semantically valid and scheduled for Phase 2.

## 7. Resolved geometry plan

The resolved plan shall record:

- plan schema and registry versions;
- DEPTH state, compressor, version, and order;
- every expanded selector key;
- implied element type;
- compressed axes;
- independent indices;
- element compressor and version;
- retained orders and physical axis options;
- shared-compressor resolution;
- materialiser adapter and materialisation version;
- implementation capability status.

The plan shall be included in canonical run configuration, telemetry, checkpoint compatibility identity, and compact identity metadata.

## 8. Console reporting

Normal systematic runs shall print each selected geometry and its implied type before training. Example:

```text
MLP_UP.MLP_HIDDEN
  implied element type:  SHEET
  compressed axes:       DEPTH × MLP_HIDDEN
  independent instances: MLP_D_MODEL
  compressor:            jpeg_like@jpeg_like_v1
```

`--explain-geometry` resolves and reports the full plan, including current implementation status and any legacy adapter, then exits before dataset access or model construction.

## 9. Existing-materialiser adapters in Phase 1

Phase 1 shall adapt, without numerical change:

1. DEPTH only → existing `depth` trajectory.
2. DEPTH plus `MLP_UP.MLP_HIDDEN` with `jpeg_like` → existing `jpeg_like_v1` trajectory.

Every other permitted selector remains represented and explainable but is rejected for training with the Phase 2 capability diagnostic.

## 10. Phase 2 preparation

Phase 1 shall provide:

- semantic element metadata rather than generic row/column assumptions;
- explicit element axes and hard attention boundaries;
- compressor capability metadata for dimensionality and physical options;
- a stable `resolved plan → capability resolution → materialiser factory` boundary;
- no requirement that future MLP projection widths equal `4 × D_MODEL` in the systematic registry design.

The current numerical model may continue to use its existing projection factor until Phase 2 replaces the relevant materialisers.

## 11. Compatibility and safety

- Legacy presets and short options shall remain numerically unchanged.
- Legacy and systematic geometry controls shall not be silently combined.
- Existing legacy BLOCK paths shall not be rewritten in Phase 1.
- Old checkpoints shall load with no systematic plan and retain their existing identity.
- New systematic checkpoints shall compare the resolved plan as model compatibility identity.
- No silent reinterpretation of checkpoint geometry is permitted.

## 12. Acceptance requirements

| ID | Requirement |
|---|---|
| UI-001 | `--select-depth`, repeatable `--select-element`, repeatable `--option`, and `--explain-geometry` exist. |
| REG-001 | The registry contains exactly the twelve non-DEPTH selectors in Section 3. |
| REG-002 | Bare MLP selectors imply SHEET and bare attention selectors imply SHEET_SET. |
| SEM-001 | `ATTENTION_HEAD` and `QKV_ROLE` are never selectable axes. |
| SEM-002 | Qualified selectors do not accumulate into a SHEET. |
| SEM-003 | Complete and qualified selectors for the same element cannot overlap. |
| CMP-001 | Phase 1 enforces one shared non-DEPTH compressor family/version. |
| CAP-001 | Registered but unimplemented geometries are explainable and fail before model allocation. |
| DIA-001 | Console output states `CURVE`, `SHEET`, or `SHEET_SET` for every selected element. |
| ID-001 | The resolved plan is persisted in run and checkpoint compatibility identity. |
| LEG-001 | Existing legacy commands retain their existing behavior. |
| MAT-001 | Existing DEPTH and JPEG-like paths are reachable through adapters without numerical changes. |
| P2-001 | The registry and capability boundary are sufficient for Phase 2 to add materialisers without changing public selector syntax. |
