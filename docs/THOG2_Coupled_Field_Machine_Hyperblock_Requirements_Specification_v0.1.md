# THOG2 Coupled Field Machine HYPERBLOCK

**Concept, Implementation and Requirements Specification**  
**Version 0.1 — 1 August 2026**  
**Status: Proposed concept, implementation and requirements specification**

> **SCOPE**  
> One architecture-wide coefficient system for the six large repeated Transformer-block matrix families; a common `WEIGHT_FAMILY × DEPTH × D_MODEL` field plus attention-specific and MLP-specific dimensions; fixed anisotropic tensor-product support; and a plug-in basis provider with Chebyshev as the first implementation.

---

## Contents

1. Executive Summary
2. Scope, Hypothesis and Governing Decisions
3. The Operational Weight Object
4. The Coupled Field Domain
5. Anisotropic Tensor-Product Expansion
6. Common and Branch-Specific Coefficient Spaces
7. Cross-Axis Relationships and Diagonals
8. Coefficient Budget and Compression Accounting
9. Interpretation of the v0 Experiment
10. Architectural Placement
11. Field-Machine and Compressor Interfaces
12. Configuration, CLI and Resolved Identity
13. Initialization and Materialization
14. Gradient Flow and Training Integration
15. Checkpoints, Diagnostics and Explain Mode
16. Test, Benchmark and Delivery Plan
17. Requirements and Acceptance Criteria
18. Acceptance Scenarios
19. Risks, Failure Modes and Deferred Work
Appendix A. Mathematical Notation
Appendix B. Canonical Example Configuration
Appendix C. Proposed Module Layout

---

# 1. Executive Summary

> **CORE DECISION**  
> HYPERBLOCK v0 will treat the six large repeated Transformer-block matrix families as one coupled field problem. It will use one fixed coefficient system with a common region over `WEIGHT_FAMILY × DEPTH × D_MODEL`, an attention extension over `ATTENTION_HEAD × ATTENTION_HEAD_CHANNEL`, and an MLP extension over `MLP_HIDDEN`. The coefficient support is fixed for the entire run; there is no breathing, pruning or regrowth in v0.

The proposed Coupled Field Machine HYPERBLOCK is an architecture-level weight parameterisation. It does not compress already-materialised weights and it does not place a second compressor downstream of DEPTH. Instead, it owns the persistent parameters for the covered matrix families and directly materialises the operational Q, K, V, attention-output, MLP-up and MLP-down weights consumed by the forward pass.

The central hypothesis is deliberately broader than Chebyshev. The hypothesis is that gradient descent can organise a Transformer so that its repeated matrix weights inhabit a compact multidimensional field. Chebyshev is the first fixed basis chosen to test that proposition because it is deterministic, already implemented in THOG2, supports anisotropic retained orders and forms tensor-product mixed terms without requiring us to nominate privileged axis pairs in advance.

The operational weights do not form one ordinary rectangular tensor. Attention weights use `DEPTH`, `D_MODEL`, `ATTENTION_HEAD` and `ATTENTION_HEAD_CHANNEL`; MLP weights use `DEPTH`, `D_MODEL` and `MLP_HIDDEN`. HYPERBLOCK therefore uses a branched mathematical domain: the attention and MLP populations share a common base and then extend along the dimensions that physically exist for each population. This is reality impinging on the object, not a weakness of the compressor.

Within each valid region, every retained tensor-product mode combination exists from the start. Gradient descent may therefore discover single-axis structure, diagonal structure and higher-order cross-axis relationships without a preselected list of special pairs. The v0 design intentionally avoids adaptive support so that the experiment tests the coupled field itself rather than a second algorithm that reallocates coefficient capacity during training.

| Decision area | Version 0.1 position |
|---|---|
| Covered weights | Q, K, V, ATTENTION_OUTPUT, MLP_UP and MLP_DOWN. |
| Persistent parameters | One structured coefficient object with common, attention-unique and MLP-unique regions. |
| Compressor | A plug-in separable axis-basis provider; `CHEBYSHEV@v1` is the first implementation. |
| Coefficient support | Fixed and fully populated inside each valid tensor-product region; no breathing in v0. |
| Existing DEPTH | Disabled for covered matrices. HYPERBLOCK owns the DEPTH dimension itself. |
| Existing BLOCK | Preserved unchanged. HYPERBLOCK is a separate architecture-wide capability. |
| Excluded initially | Linear biases, LayerNorm parameters, token/position embeddings and tied LM head. |
| Primary test | Whether a Transformer trained from scratch can organise all covered weights around one compact field coordinate system. |

---

# PART I — Concept and Mathematics

# 2. Scope, Hypothesis and Governing Decisions

## 2.1 Scope

This specification defines the mathematical object, covered operational weight population, fixed coefficient spaces, basis-provider boundary, materialisation path, configuration and identity rules, integration constraints, tests and acceptance criteria for the first Coupled Field Machine HYPERBLOCK implementation.

The first implementation is intentionally narrow in operational scope and broad in mathematical coupling. It covers only the six dominant repeated block matrix families, but couples those families through one coefficient system. It does not initially cover small vector parameters or embedding populations. That exclusion is not a claim that those parameters cannot participate later; it prevents small, structurally different populations from obscuring the first test of the central idea.

## 2.2 Central hypothesis

> **CENTRAL HYPOTHESIS**  
> When all major repeated block matrix weights are generated from one anisotropic tensor-product field, gradient descent can arrange the learned Transformer so that useful variation across `WEIGHT_FAMILY`, `DEPTH`, `D_MODEL` and the applicable attention or MLP dimensions is represented by a compact set of mixed field coefficients.

The hypothesis does not say that the present integer ordering of `D_MODEL`, `MLP_HIDDEN`, `ATTENTION_HEAD` or `ATTENTION_HEAD_CHANNEL` is naturally polynomial. It says these internal axes have enough representational freedom during training for the model to organise itself around a fixed coordinate system. That is a strong claim, but it is the claim worth testing. Omitting an axis because its initial ordering looks arbitrary would prevent the experiment from discovering whether the trained system can make that ordering useful.

Chebyshev is therefore a test vehicle, not an article of faith. A positive result would be strong evidence for the field hypothesis because the model succeeded under a generic fixed basis. A negative result would be informative but not conclusive: it could indict the basis family, retained orders, family ordering, initialization, optimization or materialisation cost rather than the architecture-level idea itself.

## 2.3 Governing decisions

1. HYPERBLOCK owns all covered matrix weights from persistent coefficients to operational materialisation. It is not composed downstream of DEPTH or another compressor.
2. The six weight families remain distinct physical populations, but family identity is exposed as one field coordinate so that cross-family structure is learnable.
3. Every physical axis that exists for a covered scalar weight is represented. No arbitrary pair or subset is declared privileged.
4. The common field is included because, without it, attention and MLP are merely two independent BLOCKs. Sharing is made available but not forced to dominate.
5. The coefficient support is fixed for a run. Full tensor-product interactions are present inside each valid region; invalid `MLP_HIDDEN × attention-axis` combinations do not exist.
6. Anisotropic retained orders are first-class configuration. DEPTH may use 16 modes without requiring every other axis to use 16.
7. The basis provider is replaceable. Field topology, coefficient ownership and materialisation must not be hard-wired to Chebyshev.
8. Existing dense, DEPTH, CURVE, SHEET and BLOCK paths remain operational and unchanged. HYPERBLOCK is opt-in and mutually exclusive with them for covered matrices in v0.

## 2.4 Terms

| Term | Meaning |
|---|---|
| Operational weight | A dense matrix consumed by nanoGPT during a forward pass. It may be ephemeral. |
| Weight family | Q, K, V, ATTENTION_OUTPUT, MLP_UP or MLP_DOWN. |
| Physical axis | An operational coordinate such as DEPTH, D_MODEL or MLP_HIDDEN. |
| Mode axis | The retained basis-index axis corresponding to one physical axis. |
| Common field | The field varying over `WEIGHT_FAMILY × DEPTH × D_MODEL`, broadcast through applicable unique dimensions. |
| Attention extension | Terms varying over `ATTENTION_HEAD` and/or `ATTENTION_HEAD_CHANNEL`. |
| MLP extension | Terms varying over `MLP_HIDDEN`. |
| Coefficient object | The learned tuple of common, attention-extension and MLP-extension coefficient arrays. |
| Field-machine | The fixed topology and contractions that evaluate coefficients into all covered weights. |
| Basis provider | A plug-in supplying one-dimensional basis tables and projection support. |
| Anisotropic | Different axes may retain different numbers of modes. |
| Non-breathing | Coefficient shapes and support are fixed for the entire run. |

# 3. The Operational Weight Object

## 3.1 Covered weight families

HYPERBLOCK v0 covers the six large matrix families repeated in every Transformer block. Q, K and V are separate families even when nanoGPT stores them in one packed projection tensor. Packing is an implementation detail; the mathematical addresses preserve the Q/K/V distinction.

| Weight family | Operational shape per layer | Scalar address |
|---|---|---|
| Q | `[D_MODEL, D_MODEL]`, viewed as `[ATTENTION_HEAD, ATTENTION_HEAD_CHANNEL, D_MODEL]` | `(Q, depth, d_model, head, head_channel)` |
| K | Same as Q | `(K, depth, d_model, head, head_channel)` |
| V | Same as Q | `(V, depth, d_model, head, head_channel)` |
| ATTENTION_OUTPUT | `[D_MODEL, D_MODEL]`, viewed as `[D_MODEL, ATTENTION_HEAD, ATTENTION_HEAD_CHANNEL]` | `(O, depth, d_model, head, head_channel)` |
| MLP_UP | `[MLP_HIDDEN, D_MODEL]` | `(UP, depth, d_model, mlp_hidden)` |
| MLP_DOWN | `[D_MODEL, MLP_HIDDEN]` | `(DOWN, depth, d_model, mlp_hidden)` |

## 3.2 Recognisable axes

| Axis | Physical length | Where it exists | Role |
|---|---:|---|---|
| WEIGHT_FAMILY | 6 | All covered weights | Selects Q, K, V, O, UP or DOWN and makes cross-family structure available. |
| DEPTH | `n_layer` | All covered weights | Selects Transformer block. This replaces separate DEPTH compression for covered matrices. |
| D_MODEL | `n_embd` | All covered weights | Model-width coordinate present once in every covered matrix family. |
| MLP_HIDDEN | `n_mlp` | MLP_UP and MLP_DOWN | Expanded MLP coordinate. |
| ATTENTION_HEAD | `n_head` | Q, K, V and O | Selects attention head. |
| ATTENTION_HEAD_CHANNEL | `n_embd / n_head` | Q, K, V and O | Selects channel within one head. |

## 3.3 One D_MODEL axis

Input and output orientation does not create a second D_MODEL axis. Q/K/V use D_MODEL on the input side and `HEAD × HEAD_CHANNEL` on the output side. ATTENTION_OUTPUT reverses that orientation. MLP_UP uses D_MODEL on the input side and MLP_HIDDEN on the output side; MLP_DOWN reverses it. Weight-family identity tells the router how to orient the generated tensor. Inventing `D_MODEL_INPUT` and `D_MODEL_OUTPUT` would duplicate one semantic coordinate and make the field more arbitrary, not less.

## 3.4 Explicit v0 exclusions

The following remain on their existing paths: all linear biases, both LayerNorm weight/bias pairs in each block, final LayerNorm, token embedding/tied LM head and position embedding. The causal mask, dropout and GELU contain no learned weights.

# 4. The Coupled Field Domain

## 4.1 Why the operational object is branched

Attention and MLP matrices share `WEIGHT_FAMILY`, `DEPTH` and `D_MODEL`, but do not share their remaining dimensions. No scalar MLP weight has an ATTENTION_HEAD coordinate. No scalar attention weight has an MLP_HIDDEN coordinate. A direct `MLP_HIDDEN × ATTENTION_HEAD` term has no operational point at which it can be evaluated.

The correct operational object is the disjoint union of two rectangular domains with a common base. Each branch is internally regular. The irregularity appears only when both valid populations are viewed together.

\[
\Omega_A=\mathcal{F}_A\times\mathcal{L}\times\mathcal{D}\times\mathcal{H}\times\mathcal{C},
\qquad
\Omega_M=\mathcal{F}_M\times\mathcal{L}\times\mathcal{D}\times\mathcal{M}.
\]

Here `𝓕_A={Q,K,V,O}`, `𝓕_M={UP,DOWN}`, `𝓛` is DEPTH, `𝓓` is D_MODEL, `𝓗` is ATTENTION_HEAD, `𝓒` is ATTENTION_HEAD_CHANNEL and `𝓜` is MLP_HIDDEN.

## 4.2 The common base

\[
\Omega_0=\mathcal{F}\times\mathcal{L}\times\mathcal{D},
\qquad
\mathcal{F}=\mathcal{F}_A\cup\mathcal{F}_M.
\]

Each branch has a natural projection to the common base:

\[
\pi_A(f,l,d,h,c)=(f,l,d),
\qquad
\pi_M(f,l,d,m)=(f,l,d).
\]

The common field is evaluated on that base and broadcast through the applicable unique dimensions. This is the precise sense in which one field-machine couples the two branches.

## 4.3 One machine rather than independent regional machines

The coefficient object has three regions because the domain has three kinds of basis term: common terms, attention-unique terms and MLP-unique terms. That does not imply three independent field-machines. Independence would mean attention and MLP had no shared parameters and could not exchange gradient information. In the proposed object, both branches evaluate the same common field and both branches contribute gradients to its coefficients.

> **HARD DISTINCTION**  
> Three storage arrays are an operational convenience. The mathematics is one direct-sum coefficient space with a shared subspace. Removing the common subspace produces two unrelated BLOCKs and ceases to test architecture-wide coupling.

# 5. Anisotropic Tensor-Product Expansion

## 5.1 Coordinate maps

Each physical index on an axis is assigned a fixed coordinate in `[-1,1]`. For an axis X of length `N_X>1`, the simplest map is:

\[
\xi_X(i)=\frac{2i}{N_X-1}-1.
\]

A singleton axis maps to zero. The family axis uses a fixed, checkpointed ordering and the same coordinate map. `CHEBYSHEV@v1` evaluates first-kind Chebyshev modes on these coordinates and may apply existing THOG2 normalization or QR stabilization. The implementation records the exact table identity used.

## 5.2 Anisotropic retained orders

Each axis has its own retained order. The plan contains:

- `P_F0`: common WEIGHT_FAMILY order, maximum 6;
- `P_FA`: attention-family order, maximum 4;
- `P_FM`: MLP-family order, maximum 2;
- `P_L`: DEPTH order;
- `P_D`: D_MODEL order;
- `P_H`: ATTENTION_HEAD order;
- `P_C`: ATTENTION_HEAD_CHANNEL order;
- `P_M`: MLP_HIDDEN order.

DEPTH order 16 was empirically useful in THOG2, but that does not imply 16 is optimal for every other axis. Orders must vary independently without changing the domain topology.

## 5.3 The common field

\[
G_0(f,l,d)=
\sum_{a=0}^{P_{F0}-1}
\sum_{b=0}^{P_L-1}
\sum_{q=0}^{P_D-1}
C^0_{abq}
T_a(\xi_F(f))
T_b(\xi_L(l))
T_q(\xi_D(d)).
\]

Every `C⁰_abq` is a learned scalar. Mixed family × depth × D_MODEL modes are present from the start.

## 5.4 The attention extension

\[
G_A(f,l,d,h,c)=
\sum_{a,b,q}
\sum_{\substack{r,s\\r+s>0}}
C^A_{abqrs}
T_a(\xi_F(f))
T_b(\xi_L(l))
T_q(\xi_D(d))
T_r(\xi_H(h))
T_s(\xi_C(c)).
\]

The condition `r+s>0` excludes the term constant along both attention-only axes. That term already belongs to the common field. Every other retained combination is available, including pure HEAD terms, pure HEAD_CHANNEL terms, HEAD × HEAD_CHANNEL terms and all interactions with family, depth and D_MODEL.

## 5.5 The MLP extension

\[
G_M(f,l,d,m)=
\sum_{a,b,q}
\sum_{t=1}^{P_M-1}
C^M_{abqt}
T_a(\xi_F(f))
T_b(\xi_L(l))
T_q(\xi_D(d))
T_t(\xi_M(m)).
\]

The MLP sum begins at `t=1` because `t=0` is constant along MLP_HIDDEN and is already represented by the common field.

## 5.6 Materialised fields

\[
W_A=G_0\circ\pi_A+G_A,
\qquad
W_M=G_0\circ\pi_M+G_M.
\]

The addition is not a second compression pass. Both terms are evaluated directly from persistent coefficients into the same operational scalar.

# 6. Common and Branch-Specific Coefficient Spaces

## 6.1 Direct-sum structure

\[
\mathcal{V}=\mathcal{V}_0\oplus\mathcal{V}_A^{+}\oplus\mathcal{V}_M^{+}.
\]

`𝓥₀` is the tensor-product space over `WEIGHT_FAMILY × DEPTH × D_MODEL`. `𝓥_A⁺` is the attention tensor-product space with the doubly constant HEAD/HEAD_CHANNEL mode removed. `𝓥_M⁺` is the MLP tensor-product space with the constant MLP_HIDDEN mode removed. The learned coefficient object is one tuple:

\[
C=(C^0,C^A,C^M).
\]

Because duplicated constant modes are excluded, the representation is identifiable at the basis level: a term cannot move arbitrarily between the common field and a branch extension without changing the generated function.

## 6.2 Why the common field is included

Without `C⁰`, the model has one attention field and one MLP field. They may use the same basis and both include DEPTH and D_MODEL, but no coefficient is shared. That is two BLOCK-like parameterisations rather than one HYPERBLOCK. Including `C⁰` lets attention and MLP losses update the same family/depth/D_MODEL structure, while branch extensions retain full freedom to add physically required variation.

The common field does not force the families to be numerically similar. WEIGHT_FAMILY remains an axis, so the common field may vary across families. If little common structure exists, common coefficients may remain small while extensions carry most useful variation.

## 6.3 Family coordinate identity

`WEIGHT_FAMILY` is not an axis inside any one matrix. It is an axis of the architecture-wide field. The canonical v0 physical order is:

```text
Q, K, V, ATTENTION_OUTPUT, MLP_UP, MLP_DOWN
```

The ordering, coordinate map and retained orders are semantic checkpoint identity. Changing them changes the meaning of every coefficient.

# 7. Cross-Axis Relationships and Diagonals

## 7.1 What gradient descent can discover

Every tensor-product coefficient multiplies one mode from each participating axis. An attention coefficient may encode a joint family × depth × D_MODEL × head × head-channel pattern. An MLP coefficient may encode a joint family × depth × D_MODEL × MLP-hidden pattern. Gradient descent updates these joint coefficients directly from the loss.

No pair list is required. A depth-only pattern uses zero-order modes on other coordinates. A DEPTH × D_MODEL diagonal uses mixed depth and D_MODEL modes. A family-specific head-channel trajectory uses family, depth and head-channel modes. Full rectangular support inside each valid region makes all retained combinations available concurrently.

## 7.2 Diagonal structure

Tensor-product bases are coordinate-aligned, but are not limited to axis-aligned functions. A dependency such as `g(x-y)` can be approximated by a sum of products `T_i(x)T_j(y)`. Curved and higher-dimensional relationships are represented by mixed products in the same way. The cost is that a thin, sharp or badly aligned diagonal may require many coefficients. “Can represent” and “will represent compactly” are different claims.

The v0 experiment deliberately accepts that risk. A learned Transformer has freedom unavailable when compressing a fixed trained model: it may arrange internal features so useful structure becomes smoother in the imposed coordinates. Success would mean not merely that Chebyshev fitted an existing ordering, but that training learned to inhabit the field.

## 7.3 What cannot exist directly

There is no direct `MLP_HIDDEN × ATTENTION_HEAD` or `MLP_HIDDEN × ATTENTION_HEAD_CHANNEL` coefficient because no operational scalar has both coordinates. The populations can still relate through shared family, depth and D_MODEL modes. Direct unique-axis coupling would require an invented shared latent coordinate or learned map and is deferred.

## 7.4 Arbitrary ordering is part of the test

The lack of a natural order on D_MODEL, MLP_HIDDEN, HEAD, HEAD_CHANNEL and WEIGHT_FAMILY is not concealed. It is the central risk. We include these axes because excluding them would make a stronger negative assumption: that useful cross-axis compression cannot emerge there.

# 8. Coefficient Budget and Compression Accounting

## 8.1 Fixed full support inside valid regions

The earlier fear of `16^10` came from imagining one dense Cartesian product over every candidate axis. The branched domain is much smaller. The common field is three-dimensional, the attention extension is five-dimensional and the MLP extension is four-dimensional. Invalid cross-branch products do not exist. Full fixed support inside each valid region is therefore feasible and preferable in v0 to sparse or adaptive support.

\[
N_C=
P_{F0}P_LP_D+
P_{FA}P_LP_D(P_HP_C-1)+
P_{FM}P_LP_D(P_M-1).
\]

The subtractions remove branch terms constant along all branch-only dimensions and therefore already present in the common field.

## 8.2 Dense-equivalent block weights

For MLP width `4D`, the six covered matrices contain `12D²` scalars per layer: four `D×D` attention matrices and two `4D×D` MLP matrices.

\[
N_W=4LD^2+2L(4D^2)=12LD^2.
\]

## 8.3 Canonical L32/D1024 example

For `L=32`, `D=1024`, `H=16`, `HEAD_CHANNEL=64` and `MLP_HIDDEN=4096`, covered dense-equivalent weights total:

\[
N_W=402,653,184.
\]

With exact family orders `6/4/2`, DEPTH order 16 and all other numerical orders 16:

| Region | Coefficients |
|---|---:|
| Common | 1,536 |
| Attention extension | 261,120 |
| MLP extension | 7,680 |
| **Total** | **270,336** |
| **Dense / coefficients** | **1,489.45×** |

Selected anisotropic examples:

| P_D | P_H | P_C | P_M | Coefficients | Dense / coefficients |
|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 8 | 8 | 34,048 | 11,826× |
| 16 | 8 | 8 | 16 | 73,728 | 5,461× |
| 16 | 16 | 16 | 16 | 270,336 | 1,489× |
| 32 | 16 | 16 | 32 | 557,056 | 723× |
| 64 | 16 | 16 | 64 | 1,179,648 | 341× |

These are parameter-count comparisons, not promises of end-to-end memory or compute reduction. Operational matrices still exist during the forward pass unless a future fused kernel consumes coefficients directly.

## 8.4 Aggressiveness

A 1,489× ratio is not evidence the model will work. It is evidence the object is compact enough to test. Viable runs may require larger D_MODEL, HEAD_CHANNEL or MLP_HIDDEN orders. The configuration must let orders rise independently without changing topology.

# 9. Interpretation of the v0 Experiment

## 9.1 What success would mean

A successful run would show that a Transformer trained from scratch can reach useful loss while all six dominant repeated matrix families are generated from one compact coupled field. Stronger evidence would include stable optimization, nontrivial common-field use, competitive token efficiency and robustness across order changes and basis providers.

## 9.2 What failure would mean

A failed run establishes that the tested configuration failed. It does not prove architecture-wide field compression impossible. At least five failure classes must be distinguished: insufficient coefficient budget, poor axis ordering, unsuitable basis family, bad initialization/conditioning and prohibitive materialisation behavior.

## 9.3 Non-breathing is methodological discipline

Adaptive growth and pruning may later be useful, but would make the first result ambiguous. HYPERBLOCK v0 keeps every shape and support set fixed. Once the static field is understood, breathing can be tested separately.

## 9.4 Baselines

Minimum comparisons are:

1. dense nanoGPT;
2. pure DEPTH with comparable persistent parameter count;
3. the strongest current BLOCK/SHEET configuration;
4. uncoupled attention-plus-MLP fields with the common field removed.

The uncoupled ablation is essential because it measures whether the common subspace contributes anything beyond two independent regional compressors.

---

# PART II — Proposed Implementation

# 10. Architectural Placement

> **IMPLEMENTATION POSITION**  
> Implement HYPERBLOCK as a new architecture-wide capability, not as another existing BLOCK selector or as a DEPTH-plus-intra composition. Reuse basis, configuration, checkpoint, materialisation and fast-discard infrastructure where contracts genuinely match.

## 10.1 Why separate from existing BLOCK

Existing BLOCK semantics are element-local. A BLOCK belongs to one projection or preserved set of instances, and DEPTH remains the outer puppeteer when enabled. HYPERBLOCK has different ownership: one coefficient object jointly owns six matrix families and includes DEPTH as an internal field axis. Forcing it through existing BLOCK selection would either lie about the geometry or create special-case machinery.

A separate subsystem protects working code paths. Dense, DEPTH, JPEG-like, CURVE, SHEET and BLOCK materializers remain untouched. HYPERBLOCK is selected explicitly and builds a different parameter ownership map for covered weights. Shared utilities may be reused, but shared names must not imply shared semantics where none exist.

## 10.2 Reuse boundaries

| Subsystem | Reuse position |
|---|---|
| Basis registry and Chebyshev implementation | Reuse through an axis-basis provider adapter. Do not fork numerical basis code. |
| Resolved configuration | Reuse serialization, validation and identity patterns; introduce `ResolvedHyperblockPlan`. |
| Operational weight injection | Reuse the established mechanism substituting generated tensors for Linear weights. |
| Fast discard false | Reuse materialize-once/project-once lifecycle, with one field-machine VJP. |
| Existing BLOCK classes | Do not subclass merely for convenience. Share low-level helpers only where contracts are exact. |
| Legacy presets and wrappers | Preserve unchanged; add explicit HYPERBLOCK controls. |

## 10.3 Mutual exclusion

When HYPERBLOCK is active, the six covered matrices must not simultaneously receive DEPTH, CURVE, SHEET, BLOCK or remainder parameterisation. Validation rejects overlap before model construction. Uncovered vectors and embeddings continue through existing paths.

# 11. Field-Machine and Compressor Interfaces

## 11.1 Proposed components

| Component | Responsibility |
|---|---|
| `ResolvedHyperblockPlan` | Immutable canonical topology, family order, axis orders, compressor identity, coordinate maps, dtype policy and materialisation version. |
| `HyperblockAxisSpec` | Axis name, physical length, retained order, coordinate map, basis options and validation. |
| `HyperblockRegionSpec` | COMMON, ATTENTION and MLP region definitions, family subsets, axis lists and excluded constant support. |
| `AxisBasisProvider` | One-dimensional basis construction, projection support, identity and diagnostics. |
| `ChebyshevAxisBasisProvider` | First provider, adapting existing stabilized Chebyshev implementation. |
| `CoupledFieldMachine` | Owns `C⁰`, `Cᴬ`, `Cᴹ`; materialises all covered weights and reports accounting. |
| `HyperblockWeightRouter` | Maps semantic generated tensors to nanoGPT `c_attn`, `c_proj`, `c_fc` layouts. |
| `HyperblockInitializer` | Projects ordinary nanoGPT target-space initialization into coupled coefficient spaces. |

## 11.2 Plug-in basis-provider contract

The topology must not call Chebyshev code directly. It asks a registered provider for a basis table for each axis. A separable provider returns `[physical_length, retained_order]`, plus stable projection and validation behavior.

```python
class AxisBasisProvider(Protocol):
    identity: BasisIdentity

    def build_basis(
        self,
        axis: HyperblockAxisSpec,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor: ...

    def project(
        self,
        values: torch.Tensor,
        axis: HyperblockAxisSpec,
        *,
        dimension: int,
    ) -> torch.Tensor: ...

    def validate(self, axis: HyperblockAxisSpec) -> list[Diagnostic]: ...
    def describe(self, axis: HyperblockAxisSpec) -> Mapping[str, Any]: ...
```

The provider may use Chebyshev, DCT, Haar, lapped cosine or another separable basis. Compressor identity includes family, version and normalization. A basis change is checkpoint-incompatible unless an explicit migration transforms coefficients.

## 11.3 Honest plug-and-play boundary

The v0 boundary supports separable one-dimensional bases whose multidimensional field is formed by tensor products. A genuinely non-separable joint compressor does not fit this interface. It would require a region-level provider or a later field-machine version. The implementation must not pretend every future compressor is plug-compatible merely by sharing a name.

## 11.4 Basis caching

Basis tables are fixed for a resolved plan and cached by device, dtype, physical length, retained order, provider version, coordinate map and options. They are buffers, not parameters. Tables with different coordinate maps or normalization must never share cache identity.

# 12. Configuration, CLI and Resolved Identity

## 12.1 Proposed canonical controls

```bash
--hyperblock coupled_field_machine
--hyperblock-compressor chebyshev@v1
--hyperblock-order WEIGHT_FAMILY_COMMON=6
--hyperblock-order WEIGHT_FAMILY_ATTENTION=4
--hyperblock-order WEIGHT_FAMILY_MLP=2
--hyperblock-order DEPTH=16
--hyperblock-order D_MODEL=16
--hyperblock-order MLP_HIDDEN=16
--hyperblock-order ATTENTION_HEAD=16
--hyperblock-order ATTENTION_HEAD_CHANNEL=16
--explain-hyperblock
```

Long-form names are recommended initially. Reusing P/Q/J/O/X/Y would be misleading because HYPERBLOCK shares one D_MODEL order across attention and MLP and one HEAD_CHANNEL order across Q/K/V/O.

## 12.2 Resolved plan fields

| Field | Required content |
|---|---|
| `schema_version` | Resolved plan format version. |
| `topology` | `COUPLED_FIELD_MACHINE@v1` and region definitions. |
| `covered_families` | Ordered list Q, K, V, O, UP, DOWN. |
| `family_coordinates` | Exact coordinate assigned to every family. |
| `axis_specs` | Physical lengths, retained orders, maps and basis options. |
| `compressor` | Registered name, version and normalization identity. |
| `coefficient_shapes` | Canonical shapes and excluded support for `C⁰`, `Cᴬ`, `Cᴹ`. |
| `materialization_version` | Operational routing/orientation identity. |
| `initialization_version` | Projection and target initialization identity. |
| `dtype_policy` | Persistent, basis, contraction and operational dtypes. |
| `compatibility` | Mutual exclusions and uncovered parameter paths. |

## 12.3 Example invocation

```bash
./train_OWT.sh \
  -g HYPERBLOCK_CFM_16 \
  --hyperblock coupled_field_machine \
  --hyperblock-compressor chebyshev@v1 \
  --hyperblock-order DEPTH=16 \
  --hyperblock-order D_MODEL=16 \
  --hyperblock-order MLP_HIDDEN=16 \
  --hyperblock-order ATTENTION_HEAD=16 \
  --hyperblock-order ATTENTION_HEAD_CHANNEL=16 \
  -L 32 -H 16 -D 1024 -C 1024 \
  --explain-hyperblock
```

The resolved plan, not preset label or short option spelling, is authoritative for checkpoints, artifact names and reproducibility.

# 13. Initialization and Materialization

## 13.1 Coefficient storage

Store only occupied regions. `c_common`, `c_attention` and `c_mlp` may be separate Parameters or entries in a ParameterDict, but serialize as one CoupledFieldMachine state. Do not allocate a zero-filled six-dimensional tensor.

| Parameter | Canonical mode shape |
|---|---|
| `c_common` | `[P_F0, P_DEPTH, P_D_MODEL]` |
| `c_attention` | `[P_FA, P_DEPTH, P_D_MODEL, P_HEAD, P_HEAD_CHANNEL]`, excluding or fixing to zero the `[HEAD=0, HEAD_CHANNEL=0]` plane |
| `c_mlp` | `[P_FM, P_DEPTH, P_D_MODEL, P_MLP_HIDDEN]`, excluding or fixing to zero MLP_HIDDEN mode 0 |

## 13.2 Target-space initialization

Naively sampling coefficients independently can produce badly scaled and correlated operational weights. Preferred initialization is target-space projection:

1. Generate Q, K, V, O, MLP_UP and MLP_DOWN dense targets using current nanoGPT initialization, including residual-projection scaling where applicable.
2. For each family, average the target over its unique dimensions: HEAD × HEAD_CHANNEL for attention or MLP_HIDDEN for MLP.
3. Jointly project those six family/depth/D_MODEL means into the common basis to initialize `C⁰`.
4. Broadcast reconstructed common field, subtract it from each target population, and project remaining attention and MLP residuals into branch bases.
5. Discard temporary dense targets and report reconstruction error by family.

If basis tables are orthonormal, projection is a sequence of transposed basis contractions. Otherwise the provider supplies a stable least-squares or pseudoinverse projection. Initialization is deterministic under seed and versioned in identity.

## 13.3 Materialisation contractions

```python
common = contract(
    c_common,
    family_common_basis,
    depth_basis,
    d_model_basis,
)

attention_extra = contract(
    c_attention,
    family_attention_basis,
    depth_basis,
    d_model_basis,
    head_basis,
    head_channel_basis,
)

mlp_extra = contract(
    c_mlp,
    family_mlp_basis,
    depth_basis,
    d_model_basis,
    mlp_hidden_basis,
)

attention_weights = common[Q:O, ..., None, None] + attention_extra
mlp_weights = common[UP:DOWN, ..., None] + mlp_extra
```

## 13.4 Orientation and packed QKV

| Family | Generated semantic layout | nanoGPT operational layout |
|---|---|---|
| Q/K/V | `[layer, d_model, head, head_channel]` | Transpose/reshape to each `[D_MODEL,D_MODEL]`, then pack in current `c_attn` convention. |
| ATTENTION_OUTPUT | `[layer, d_model, head, head_channel]` | Reshape head × channel to input D_MODEL side of `c_proj`. |
| MLP_UP | `[layer, d_model, mlp_hidden]` | Orient as `[MLP_HIDDEN,D_MODEL]` for `c_fc`. |
| MLP_DOWN | `[layer, d_model, mlp_hidden]` | Orient as `[D_MODEL,MLP_HIDDEN]` for `c_proj`. |

## 13.5 No address table

Do not build one coordinate record per operational scalar. Address notation is mathematical. Operational code uses region-level basis matrices, contractions, family slices and reshapes.

# 14. Gradient Flow and Training Integration

## 14.1 Ordinary autograd reference

First implement all contractions inside autograd. Loss gradients flow from each operational weight through its extension and into the common field. Attention and MLP gradients accumulate naturally into `c_common`. This CPU path is the correctness reference even if it is not the final performance path.

## 14.2 Fast discard false

The optimized THOG2 path materialises generated operational weights once per optimizer update, detaches them for gradient-accumulation microsteps, then projects accumulated operational gradients once back to persistent parameters. CoupledFieldMachine integrates at this lifecycle boundary rather than materialising independently per layer or microstep.

The projection is the vector-Jacobian product of the common and branch contractions. Q/K/V/O gradients contribute to `c_attention` and `c_common`; MLP_UP/DOWN gradients contribute to `c_mlp` and `c_common`. Common projection sums both populations before DDP synchronization and optimizer update.

## 14.3 Gradient accumulation and DDP

Operational gradients accumulate across microsteps; coefficient projection occurs once per optimizer update. In DDP, either operational gradients are reduced before projection or projected coefficient gradients after projection, but the chosen path must be mathematically equivalent and tested. Existing manual all-reduce behavior for compact projected gradients is the likely reuse point.

## 14.4 Compile and dtype

Persistent coefficient dtype, basis dtype, contraction accumulation dtype and operational output dtype are distinct policy fields. bf16/fp16 operation may still require fp32 basis generation or contraction accumulation. Fixed non-breathing shapes must avoid per-update recompilation.

## 14.5 Regularization and optimizer

Weight decay applies to persistent coefficients, not materialised weights, unless an explicit operational regularizer is later added. That changes regularization geometry compared with dense AdamW and must be recorded. Optimizer groups should expose common and branch coefficients separately for diagnostics while using one optimizer family by default.

# 15. Checkpoints, Diagnostics and Explain Mode

## 15.1 Checkpoint contents

Checkpoint the resolved plan, coefficient tensors, optimizer state, provider identity, materialisation version, initialization version and ordinary training state. Basis tables may be regenerated if generation is deterministic and versioned. Family order must be explicit.

## 15.2 Compatibility

Reject changes in topology, family order, coordinate map, retained orders, basis normalization, materialisation orientation or excluded support unless a named migration exists. Shape-only loading is unsafe because equal shapes may carry different semantics.

## 15.3 Explain mode

```text
HYPERBLOCK
  topology: COUPLED_FIELD_MACHINE@v1
  compressor: CHEBYSHEV@v1
  covered families: Q, K, V, ATTENTION_OUTPUT, MLP_UP, MLP_DOWN
  common axes: WEIGHT_FAMILY × DEPTH × D_MODEL
  attention extension: ATTENTION_HEAD × ATTENTION_HEAD_CHANNEL
  MLP extension: MLP_HIDDEN
  coefficient support: fixed; full rectangular support in each valid region
  coefficient scalars: 270,336
  materialised scalars: 402,653,184
  dense / coefficients: 1,489.45×
  existing DEPTH/BLOCK: inactive for covered families
  capability: implemented
```

## 15.4 Runtime diagnostics

| Diagnostic | Required report |
|---|---|
| Coefficient accounting | Per-region and total scalars, bytes by dtype, dense-equivalent ratio. |
| Initialization reconstruction | RMS and maximum error per family. |
| Gradient norms | Common, attention and MLP norms separately; optional mode summaries. |
| Common-field utilization | Norm/energy attributable to common field versus extensions. |
| Materialisation timing | Common, attention, MLP and routing time per update. |
| Temporary memory | Peak memory for bases, intermediates and operational weights. |
| Conditioning | Provider-reported orthogonality/condition diagnostics. |

# 16. Test, Benchmark and Delivery Plan

## 16.1 CPU mathematical tests

1. Coefficient-count formulas match allocations for multiple anisotropic orders.
2. Common zero-order term appears exactly once; branch constant duplicates are absent or zero.
3. Grouped contractions match a slow scalar reference evaluator.
4. Every valid mixed mode affects expected coordinates; invalid MLP × attention combinations cannot be constructed.
5. Finite-difference gradients agree with autograd for all three parameter regions.
6. Attention and MLP both contribute to `c_common`; the joint gradient equals the sum of separate contributions.
7. Generated orientations reproduce expected nanoGPT QKV packing and MLP shapes.
8. A mock second basis provider replaces Chebyshev without topology or router changes.

## 16.2 Integration tests

- one-step and multi-step CPU training;
- checkpoint save/load and exact continuation;
- explicit failure on changed family ordering, order, provider or materialisation version;
- gradient-accumulation equivalence;
- fast-discard-false equivalence to ordinary autograd;
- eager, compile, single-GPU and DDP smoke tests;
- no regression in dense, DEPTH, JPEG-like or BLOCK suites.

## 16.3 Benchmark ladder

| Stage | Purpose | Minimum run |
|---|---|---|
| A. Structural smoke | Shapes, forward/backward and checkpointing. | Tiny CPU model, 2–10 updates. |
| B. Optimization smoke | Detect bad initialization or gradients. | Small OWT model, 100–500 updates. |
| C. Order sweep | Find viable anisotropic orders. | Matched 1,000–3,000 update grid. |
| D. Coupling ablation | Compare coupled field with no-common topology. | Matched persistent count and tokens. |
| E. Basis ablation | Determine Chebyshev dependence. | Chebyshev versus one compatible provider. |
| F. Long run | Assess loss, token efficiency and stability. | Only after short-run viability. |

## 16.4 Delivery stages

| Stage | Deliverable |
|---|---|
| 0. Specification | This document and resolved decisions. |
| 1. Plan and basis boundary | Plan, axis/region specs, provider protocol, Chebyshev adapter, explain-only output. |
| 2. CPU reference machine | Allocation, reference materialisation, initialization and mathematical tests. |
| 3. nanoGPT routing | Replace only six covered operational matrices; preserve all other paths. |
| 4. Training lifecycle | Autograd training, checkpointing, optimizer integration and CPU smoke. |
| 5. Optimized lifecycle | Fast discard false, compile and DDP integration. |
| 6. Benchmarks | Order sweep, uncoupled ablation and first long run. |

---

# PART III — Requirements Specification

# 17. Requirements and Acceptance Criteria

## 17.1 Concept and semantic requirements

| ID | Requirement |
|---|---|
| CON-001 | HYPERBLOCK shall own persistent representation and materialisation of Q, K, V, ATTENTION_OUTPUT, MLP_UP and MLP_DOWN when active. |
| CON-002 | The six families shall be represented within one coupled coefficient system, not six independent compressors. |
| CON-003 | The coefficient system shall contain a common `WEIGHT_FAMILY × DEPTH × D_MODEL` subspace shared by attention and MLP. |
| CON-004 | Attention-only variation shall use ATTENTION_HEAD and ATTENTION_HEAD_CHANNEL; MLP-only variation shall use MLP_HIDDEN. |
| CON-005 | No coefficient shall require MLP_HIDDEN and an attention-only coordinate simultaneously. |
| CON-006 | v0 coefficient support shall remain fixed for the entire run. |
| CON-007 | Every retained tensor-product combination inside each valid region shall be learnable; no privileged pair list is required. |
| CON-008 | Existing DEPTH and BLOCK semantics shall not be redefined to disguise HYPERBLOCK. |

## 17.2 Mathematical requirements

| ID | Requirement |
|---|---|
| MTH-001 | Operational domain shall use the attention and MLP domains defined here with explicit projection to the common base. |
| MTH-002 | Common, attention and MLP coefficient spaces shall form a direct sum with duplicated constant unique-axis modes excluded. |
| MTH-003 | Each axis shall have an independent retained order subject to physical-length and provider constraints. |
| MTH-004 | Implementation shall report coefficient counts using the canonical formula and verify against allocated parameters. |
| MTH-005 | Family order and axis coordinate maps shall be fixed, explicit and checkpointed. |
| MTH-006 | Materialised field shall equal common field plus applicable branch extension at every covered coordinate. |

## 17.3 Compressor requirements

| ID | Requirement |
|---|---|
| CMP-001 | Field topology shall consume a registered axis-basis provider rather than calling Chebyshev directly. |
| CMP-002 | `CHEBYSHEV@v1` shall be first and reuse existing stabilized THOG2 implementation where compatible. |
| CMP-003 | Provider shall expose basis construction, validation, projection support, identity and diagnostics. |
| CMP-004 | Provider identity shall include numerical version and normalization/orthogonalization convention. |
| CMP-005 | v0 provider interface shall honestly support separable axis bases; non-separable region compressors shall be rejected. |
| CMP-006 | A mock or second provider shall pass the same field-machine tests without topology changes. |

## 17.4 Implementation requirements

| ID | Requirement |
|---|---|
| IMP-001 | HYPERBLOCK shall be a separate opt-in subsystem preserving dense, DEPTH, CURVE, SHEET and BLOCK paths. |
| IMP-002 | Allocate only occupied coefficient regions; do not allocate a zero-filled six-dimensional tensor. |
| IMP-003 | Use grouped tensor contractions; do not construct a per-scalar address table. |
| IMP-004 | Q/K/V and attention-head boundaries shall remain explicit through generation and routing. |
| IMP-005 | MLP_UP and MLP_DOWN shall use the same MLP_HIDDEN semantic axis despite opposite orientation. |
| IMP-006 | Covered families shall be mutually exclusive with existing generated-weight compressors while HYPERBLOCK is active. |
| IMP-007 | Uncovered parameters shall continue through existing paths. |
| IMP-008 | New nanoGPT code shall follow THOG marking convention; existing nanoGPT lines shall not be deleted. |

## 17.5 Initialization and training requirements

| ID | Requirement |
|---|---|
| TRN-001 | Initialization shall occur in target weight space and be projected into coupled coefficient spaces, or demonstrate equivalent scaling. |
| TRN-002 | Initialization shall be deterministic under seed and versioned. |
| TRN-003 | Attention and MLP gradient flow shall both contribute to `c_common`. |
| TRN-004 | Optimized path shall project operational gradients once per optimizer update. |
| TRN-005 | DDP synchronization shall match single-process reference coefficient gradients. |
| TRN-006 | Fixed coefficient shapes shall compile without per-update graph recompilation. |
| TRN-007 | Persistent, basis, contraction and operational dtypes shall be explicit. |

## 17.6 Configuration, identity and diagnostics

| ID | Requirement |
|---|---|
| ID-001 | Canonical `ResolvedHyperblockPlan` shall be persisted in checkpoints and run metadata. |
| ID-002 | Topology, family order, maps, orders, provider, materialisation and initialization versions shall participate in identity. |
| CLI-001 | CLI shall provide explicit HYPERBLOCK selection, compressor selection and repeatable long-form axis-order controls. |
| CLI-002 | v0 CLI shall not silently reinterpret P/Q/J/O/X/Y. |
| VAL-001 | Validation shall reject overlap with existing compressors before model construction. |
| VAL-002 | Validation shall reject retained orders exceeding physical length or provider capability. |
| VAL-003 | Validation shall reject unsupported non-separable providers with a specific diagnostic. |
| DIA-001 | `--explain-hyperblock` shall resolve, account and report without starting training. |
| DIA-002 | Runtime shall report per-region coefficient count, gradient norms, materialisation timing and common utilization. |

## 17.7 Testing requirements

| ID | Requirement |
|---|---|
| TST-001 | A slow scalar reference evaluator shall exist for tiny configurations. |
| TST-002 | Grouped materialisation shall match the reference for all six families. |
| TST-003 | Finite-difference/autograd gradient tests shall cover common and both extensions. |
| TST-004 | Tests shall prove attention and MLP gradient contributions sum correctly in `c_common`. |
| TST-005 | Checkpoint compatibility tests shall cover every semantic identity field. |
| TST-006 | Existing dense, DEPTH, JPEG-like and BLOCK tests shall remain green. |
| TST-007 | CPU smoke shall complete before any GPU optimization run. |
| TST-008 | Fast-discard-false path shall be tested against ordinary autograd. |

# 18. Acceptance Scenarios

| Scenario | Pass condition |
|---|---|
| Explain-only L32/D1024 plan | Resolves six families, three regions, all orders and exact 270,336 coefficient count without constructing a training batch. |
| CPU scalar equivalence | Grouped materialisation matches reference evaluator for random tiny coefficients and all six families. |
| Common-gradient coupling | Attention loss changes `c_common`; MLP loss changes `c_common`; joint gradient equals sum of separate gradients. |
| No-common ablation | Same model can disable `c_common` for comparison and identity records uncoupled topology. |
| Provider substitution | A second compatible provider runs the tiny model without field-machine changes. |
| Checkpoint safety | Changed family order or axis order gives a specific incompatibility before loading tensors. |
| Fast-discard equivalence | One optimizer update matches ordinary autograd within tolerance under matched inputs and seed. |
| Legacy safety | Representative dense, DEPTH, JPEG-like and BLOCK smoke tests remain unchanged. |
| First optimization smoke | Small OWT model runs at least 500 updates without non-finite loss, gradient collapse or repeated compile failure. |
| Coupling evidence | Run reports common utilization and supports a matched uncoupled comparison; no conclusion is drawn from loss alone. |

# 19. Risks, Failure Modes and Deferred Work

## 19.1 Principal risks

| Risk | Why it matters | Required response |
|---|---|---|
| Arbitrary axis order | Useful functions may be spectrally rough and require high order. | Measure it; inspect order sweeps and basis alternatives rather than hiding the axis. |
| Excessive compression | First 16-mode configuration may be below viable capacity. | Increase orders anisotropically and compare persistent-count-matched baselines. |
| Common-field underuse | Model may behave like separate attention and MLP fields. | Measure common utilization and run no-common ablation. |
| Common-field interference | Shared coefficients may create gradient conflict. | Report per-branch common-gradient contributions; consider controlled scale only later. |
| Materialisation cost | High parameter compression may still be too slow. | Benchmark regions, optimize contraction order and preserve materialize-once semantics. |
| Bad initialization | Poor scaling may kill the experiment before structure emerges. | Use target-space projection and report reconstruction/variance. |
| Provider overclaim | Nominal plug-in API may not fit non-separable compressors. | Restrict v0 honestly to separable providers. |

## 19.2 Explicitly deferred

- breathing, adaptive support, pruning, regrowth or dynamic order changes;
- learned axis reordering, permutations or latent coordinate maps;
- direct MLP_HIDDEN × attention-axis coupling through an invented shared latent dimension;
- token embedding, tied LM head, position embedding, LayerNorm and bias inclusion;
- remainder/precision augmentation on HYPERBLOCK;
- non-separable region-level compressors and learned hypernetwork decoders;
- fused kernels consuming coefficients without dense materialisation;
- automatic migration from DEPTH/BLOCK checkpoints;
- per-axis compressor families inside one HYPERBLOCK plan.

## 19.3 Decision threshold

Do not proceed directly from specification to a long GPU run. CPU reference evaluation, gradient tests, initialization reconstruction and explain accounting are mandatory gates. The first GPU smoke is structural. Only a stable short-run order sweep can justify a long comparison.

> **SPECIFICATION ENDPOINT**  
> Version 0.1 stops at a fixed, separable, anisotropic coupled field over the six repeated block matrix families. It deliberately establishes one honest object, one plug-in boundary and one testable hypothesis before adaptive capacity or broader parameter coverage.

---

# Appendix A. Mathematical Notation

| Symbol | Meaning |
|---|---|
| `𝓕`, `𝓕_A`, `𝓕_M` | All weight families; attention families; MLP families. |
| `𝓛` | DEPTH coordinate set. |
| `𝓓` | D_MODEL coordinate set. |
| `𝓗` | ATTENTION_HEAD coordinate set. |
| `𝓒` | ATTENTION_HEAD_CHANNEL coordinate set. |
| `𝓜` | MLP_HIDDEN coordinate set. |
| `Ω_A`, `Ω_M` | Attention and MLP operational domains. |
| `Ω₀` | Common family × depth × D_MODEL base. |
| `C⁰`, `Cᴬ`, `Cᴹ` | Learned coefficient arrays. |
| `G₀`, `G_A`, `G_M` | Fields evaluated from coefficient arrays. |
| `W_A`, `W_M` | Materialised attention and MLP fields. |
| `P_X` | Retained order for axis/region X. |
| `ξ_X` | Fixed map from integer axis coordinate to `[-1,1]`. |
| `T_p` | First-kind Chebyshev mode p; implementation may use stabilized table. |

# Appendix B. Canonical Example Configuration

| Model quantity | Value |
|---|---:|
| n_layer / DEPTH length | 32 |
| n_embd / D_MODEL length | 1024 |
| n_head / ATTENTION_HEAD length | 16 |
| ATTENTION_HEAD_CHANNEL length | 64 |
| MLP_HIDDEN length | 4096 |
| Covered dense-equivalent scalars | 402,653,184 |

| Retained order | Value |
|---|---:|
| WEIGHT_FAMILY common / attention / MLP | 6 / 4 / 2 |
| DEPTH | 16 |
| D_MODEL | 16 |
| ATTENTION_HEAD | 16 |
| ATTENTION_HEAD_CHANNEL | 16 |
| MLP_HIDDEN | 16 |

| Coefficient region | Scalars |
|---|---:|
| Common | 1,536 |
| Attention extension | 261,120 |
| MLP extension | 7,680 |
| Total | 270,336 |
| Dense / coefficients | 1,489.45× |

This configuration is an accounting and smoke-test reference, not a claim that order 16 is optimal on every axis. The first sweep varies D_MODEL, HEAD, HEAD_CHANNEL and MLP_HIDDEN independently while keeping DEPTH at the empirically useful starting point of 16.

# Appendix C. Proposed Module Layout

```text
sheet/hyperblock/
    __init__.py
    plan.py
    topology.py
    axis_basis.py
    chebyshev_axis_basis.py
    coupled_field_machine.py
    initializer.py
    materializer.py
    weight_router.py
    diagnostics.py

tests/
    test_hyperblock_plan.py
    test_hyperblock_reference_materializer.py
    test_hyperblock_gradients.py
    test_hyperblock_initialization.py
    test_hyperblock_checkpoint_identity.py
    test_hyperblock_fast_discard.py
    test_hyperblock_cpu_smoke.py
```

Exact filenames may change, but topology, basis provider, coefficient ownership, initialization and nanoGPT routing should remain separate. The field machine must not become another monolithic trajectory class with compressor-specific branches embedded throughout it.

**SPECIFICATION END**
