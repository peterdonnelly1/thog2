# THOG2 Basis-Family Plug-in Refactor — Test Plan v0.1

## 1. Purpose

This stage restructures the existing Chebyshev and DCT basis implementations behind a single basis-family registry and one module per basis family.

The stage is structural. It must not change any existing basis matrix, model geometry, parameter count, checkpoint identity, run name, training result, or materialisation result.

## 2. Scope

The implementation shall provide:

- one common basis-family protocol;
- one registry that owns family names, aliases, versions, artifact tags, capabilities, and kernels;
- one module for Chebyshev;
- one module for DCT;
- compatibility exports for existing imports;
- registry-derived Python CLI basis choices;
- no geometry-specific or materialiser-specific changes for a new fixed basis family.

Automatic filesystem discovery is explicitly out of scope. A new family shall require one family module and one explicit registry entry.

## 3. Required regression tests

### 3.1 Exact Chebyshev preservation

- Existing Stage 0 fixture basis hashes shall remain byte-for-byte unchanged.
- Existing selected materialisations shall remain within their current exact tolerances.
- Chebyshev aliases and version aliases shall resolve to the same registered definition.
- One-point, endpoint, recurrence, deterministic-QR, and orthonormality behaviour shall remain unchanged.

### 3.2 Exact DCT preservation

- Existing DCT-II orthonormal basis values shall remain unchanged.
- Existing CPU and available CUDA tests shall pass.
- Existing tiny training runs for depth, head-aware block, MLP block, and full block shall pass.
- Existing DCT checkpoint save, resume, identity, and cross-family rejection tests shall pass.

### 3.3 Registry contract

- The registry shall return canonical family names, aliases, versions, artifact tags, metadata, and kernels.
- Duplicate family names shall fail.
- Duplicate aliases shall fail.
- Duplicate versions shall fail.
- Duplicate artifact tags shall fail.
- Unknown families shall fail clearly.
- A local test registry shall accept a third synthetic fixed basis family without changes to geometry or materialisation code.

### 3.4 Cache and ownership

- Basis cache keys shall continue to include family, version, geometry, dtype, and device.
- Basis buffers shall remain non-persistent and non-trainable.
- Cache separation between families shall remain intact.

### 3.5 Configuration and identity

- Registered basis families shall be accepted by selector validation without adding family-specific branches.
- Basis-version normalisation shall be registry-driven.
- Existing Chebyshev and DCT compact identities shall remain unchanged.
- Existing checkpoint compatibility rules shall remain unchanged.
- Existing artifact fragments shall remain `CHEBY` and `DCT`.

### 3.6 CLI and wrappers

- `run_thog2_owt.py --help` basis choices shall come from the registry.
- The scruffy and dreedle wrappers shall not contain a hardcoded `chebyshev|dct` allow-list.
- Existing wrapper shell-syntax and dry-run tests shall pass.
- Invalid basis-family values shall still fail before training begins.

### 3.7 Full regression

Run:

```bash
python -m unittest discover tests
```

Also run focused suites covering:

```bash
python -m unittest \
  tests.test_stage2_basis_kernel \
  tests.test_stage7_dct_basis_kernel \
  tests.test_stage7_dct_training_and_checkpoint \
  tests.test_stage7_dct_gpu \
  tests.test_stage8_wrapper_shell_syntax \
  tests.test_stage8_wrappers_and_run_config
```

CUDA-only tests may skip where CUDA is unavailable. Any non-CUDA failure is blocking.

## 4. Acceptance criteria

The stage is accepted only when:

1. all existing tests pass;
2. new registry-contract tests pass;
3. Chebyshev fixture hashes are unchanged;
4. DCT numerical results are unchanged;
5. checkpoint identities and artifact labels are unchanged;
6. a synthetic third basis can be registered and built without editing geometry or materialiser code;
7. adding the next real basis requires one new basis module and one explicit registry entry, plus tests for that basis only.
