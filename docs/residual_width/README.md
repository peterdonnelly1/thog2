# Residual stream spectral reparameterisation

The attached **THOG Enhancement – Residual Stream Spectral Reparameterisation** is the authoritative specification for `residuals_compression`. Its unchanged source is [THOG_Enhancement__Residual_Stream_Reparameterisation_v0.2.docx](THOG_Enhancement__Residual_Stream_Reparameterisation_v0.2.docx). The attachment filename says v0.2; the document's own title, status and requirement prose still say v0.1. This implementation follows all 142 numbered requirements in that attachment, rather than treating either label as a different scope. Source SHA-256: `e02b9ba583e20b88d8d1e35c5c3b3df7bc86ba76bf6006b763573f9aa7bd5dd5`.

See [acceptance.md](acceptance.md) for measurements, baseline regressions and qualification limits, and [requirements.csv](requirements.csv) for the complete requirement traceability matrix.

## Select width compression

Use the existing public entry point. This example has reference width D=1024, residual coefficient count r=128, attention width A=1024 and MLP hidden width F=4096. It selects width independently of depth:

```bash
python -m run_thog2_owt \
  --model-type sheet --geometry-preset width-type-I --select-width \
  --n-layer 12 --n-head 16 --n-embd 1024 \
  --option WIDTH.order=128 --option WIDTH.compressor=chebyshev \
  --explain-geometry
```

`--explain-geometry` resolves and validates without training. Remove it and supply your existing data, batch, checkpoint and instrumentation arguments to train. The example certifies a basis construction, not model quality or hardware capacity. r is a coefficient count, including mode zero. It need not be divisible by the head count; A must be divisible by it.

To add fixed depth P=6 without changing the width basis:

```bash
python -m run_thog2_owt \
  --model-type sheet --geometry-preset width-type-I --select-width --select-depth \
  --n-layer 12 --n-head 16 --n-embd 1024 \
  --option WIDTH.order=128 --option WIDTH.compressor=chebyshev \
  --option DEPTH.order=6 --option DEPTH.compressor=dct \
  --explain-geometry
```

WIDTH and DEPTH each accept `order`, `compressor` and `compressor_version`. `compressor` also accepts `family@exact_version`. Families use the existing aliases and registry: Chebyshev, DCT, Haar and lapped cosine. Width defaults to Chebyshev and version `auto`; the resolved exact version is persisted. Duplicate assignments, unknown properties, conflicting versions and unsupported requests fail before model allocation. In width mode, explicitly supplied legacy `--basis-family`/`--basis-version` (or shell `-B`/`-v`) are rejected, including explicit defaults. Use scoped options instead.

The existing `train_OWT.sh` wrapper forwards the same selector, scoped options and capture namespace. Width rejects PLASTIC, HYPERBLOCK, additional element geometries, PREMAT and materialized-weight replay. The standard AdamW, SGD, Nesterov, Adafactor and RMSprop update paths were exercised. THOGOPT's existing reconstructed-layer optimizer is not qualified for direct width contractions.

## Start a single run or grid in Runner

After updating the branch, restart the Instra backend to reload the catalogue, and hard-refresh the browser. Open **Runner → Recipes → Add Grid Recipe**. Use **Geometry** or the parameter search to set these fields:

| Field | Width-only smoke setting |
| --- | --- |
| `--geometry-preset` | `width` or `width-type-I` |
| `--select-width` | Enabled automatically by Runner; visible in Geometry |
| `WIDTH.order` | `64` |
| `WIDTH.compressor` | `dct` |
| `WIDTH.compressor_version` | Blank (resolves `auto`) |
| `DEPTH.order` | Clear the inherited default; leaving a value selects joint depth compression |
| `DEPTH.compressor`, `DEPTH.compressor_version` | Inactive when neither `DEPTH.order` nor `--select-depth` selects depth |
| `--n-embd`, `--n-layer`, `--n-head` | `256`, `4`, `4` |
| `--block-size`, `--batch-size`, `--gradient-accumulation-steps` | `128`, `1`, `1` |
| `--checkpoint-segment-size` | `2` |
| `--max-iters`, `--warmup-iters` | `20`, `0` |
| `--learning-rate`, `--min-lr` | `0.0006`, `0.00006` |
| `--eval-interval`, `--eval-iters`, `--log-interval`, `--checkpoint-interval` | `5`, `2`, `1`, `5` |
| `--optimizer`, `--premat` | `adamw`, `disabled` |
| `--data-dir` | `data/openwebtext` (prepared dataset) |
| Profiling mode | `None` |

Clear any inherited global `--basis-family`/`--basis-version`, extra attention/MLP geometry, PLASTIC or HYPERBLOCK settings. Select the intended host/GPU; its execution profile supplies precision and attention backend. For the first Runner smoke, use the profile matching the successful direct smoke: float32 and SDPA. Set optional export and checkpoint destinations as usual.

One value in every dimension gives a single run. Enter `32, 64, 128` in `WIDTH.order` for three width-only runs. To form a joint grid, also enter `2, 4` in `DEPTH.order` and `chebyshev` in `DEPTH.compressor`: the Cartesian product has six runs. A nonblank `DEPTH.order` automatically forwards `--select-depth`. Reference width, layer count and the existing numeric dimensions can also be swept. Each combination is validated before scheduling; `2 <= r <= D` and basis rank validity still apply.

Width and depth compressor families/versions are fixed Recipe fields. Create separate Recipes to compare basis families. Do not enter comma-separated family names in these fields.

To display the three width charts, use **Width Activation Curves** to set `--instrumentation__width_activation_curves__mode` to `probes`, `log_every_n_steps` to `5`, `probe_every_n_steps` to `10`, `history_length` to `5`, `sample_tokens_per_layer` to `2`, and `feature_evaluation_points` to `32`, with the full namespace prefix on every field. The default `end_step=-1` means an unbounded capture window and is accepted when entered explicitly. Without capture, the training run remains valid and ordinary loss logging continues.

Click **Save → Preview**, verify one, three or six logical runs and the intended GPU placement, then **Launch**. The Runner editor permits width-only recipes without `DEPTH.order`; legacy depth recipes still require it. See the [Runner follow-up evidence](../../evidence/residual_width_runner_followup.json) for regression and browser checks.

## Representation and normalization

Learned token and position tables are V×r and T×r. Embeddings learn coefficient coordinates directly through next-token cross-entropy. No larger learned embedding is analysed on each lookup. The basis C is a fixed r×D matrix; analysis is Cx, synthesis is Cᵀc, and CᵀCx is a lossy projection when r<D. Synthesis-gradient transport uses its adjoint C. These transforms are linear in activation values. Chebyshev polynomial recurrence determines the fixed basis entries on the registered uniform grid; deterministic QR determines the actual orthonormal coefficient coordinates.

Each block stores output-by-input operators 3D×r, r×D, 4D×r and r×4D. Residuals and residual additions remain r wide; causal QKV and ordinary GELU remain A=D and F=4D wide. The tied vocabulary head maps r directly to V. Joint depth stores these banks as [output,input,P] and accumulates their contractions using the independent fixed depth basis. It does not reconstruct a layer matrix. The existing depth-vector participation flag controls block biases and block LayerNorm affines; ln_f remains outside sampled depth.

At both block pre-norm sites and ln_f, normalization is exactly C LN_D(Cᵀc). Each site owns D-vector gain γ and optional D-vector bias β, sharing one C. The folded implementation constructs G=(C*γ)Cᵀ by column scaling and b=Cβ, with no D×D diagonal. For a positive constant first mode, centering zeros c₀ and population variance is the squared sum of other coefficients divided by D. Other orthonormal adapters use u=C1, mean=u·c/D, variance=max(‖c‖²/D−mean²,0), and numerator Gc−mean(Cγ). Epsilon is 10⁻⁵. Statistics and gain accumulation use at least float32 with autocast explicitly disabled; gradients propagate through construction. G and b are rebuilt each forward, so parameter/device/precision changes cannot leave stale derived caches.

Post-affine projection can discard represented feature components. Basis-dependent coupling is constrained by C and γ; G is not a free learned matrix. The standard pre-attention, pre-MLP and final normalization boundaries are preserved. Their mathematical equivalence is tested; optimal placement and broad language-model quality advantages remain unproven.

Raw basis construction uses float64 numerical-rank tolerance `eps_float64 * max(D,r) * sigma_max`, and checks both orthogonality and preservation of the requested raw span. Chebyshev D=1024,r=128 is resolved, with raw condition approximately 111. Its r=512 request is rejected as rank deficient; DCT provides a resolved half-width alternative. No grid, family or version is silently substituted. A nonorthonormal adapter fails the width capability contract.

## Capture controls

Every suffix below follows the exact prefix `--instrumentation__width_activation_curves__`:

| Suffix | Default | Domain / behavior |
| --- | --- | --- |
| `mode` | `off` | `off`, `basic`, `probes` |
| `log_every_n_steps` | 100 | Positive completed-update interval |
| `probe_every_n_steps` | 1000 | Positive completed-update interval |
| `history_length` | 20 | Positive maximum retained records |
| `sample_tokens_per_layer` | 8 | Positive token sample bound per residual site |
| `feature_evaluation_points` | 256 | Integer ≥2; clipped to D |
| `probe_orders` | `auto` | Distinct comma-separated integers 1≤k<r; auto=max(1,floor(r/2)) |
| `start_step` | 0 | Nonnegative inclusive completed-update boundary |
| `end_step` | -1 | -1 unbounded, otherwise ≥start_step inclusive |

Cadences are anchored at start_step. There is no step-zero optimizer update. Invalid values are rejected even when capture is off. Off allocates no width capture object, hooks or rolling sample buffers and leaves ordinary loss/run logging active.

Basic mode records mean-squared coefficient energies at `embedding.residual`, `block_i.residual_output` and `ln_f.head_input`, reference mean and centered energy, and gain variation/coupling at every `block_i.ln_1`, `block_i.ln_2` and `ln_f`. Records include location, actual operation token count N, dimensions, sampled count and memory windows.

Probes evaluate the unmasked trained model and each retained-prefix ablation on the same fixed validation tokens. Masks apply at embeddings, normalization inputs/outputs, branch outputs, residual additions and final head input; model shapes and parameter identities stay fixed. Positive delta loss means the ablation harmed this trained model. It does not estimate a separately retrained smaller model. Independent sampling state, token count, baseline loss and duration are persisted; optimizer parameters, gradients, training RNG and module modes are preserved. Resume restores the recorded sample selection.

Feature curves synthesize only bounded token samples and configured grid points. Normalization diagnostics calculate actual post-affine projection-discarded energy in bounded 256-feature tiles, rather than using the identity CCᵀc as a quality metric. Histories are detached and bounded. Capture reports its elapsed time, retained Python-buffer bytes and serialized history bytes separately. A bounded profiler distinguishes normalization construction from application without normal per-site synchronization.

Width records use the existing chart store and configured W&B export. Instra Runs and Multiview show mode energy, retained-prefix loss and feature curves with step/site selectors, existing maximize/restore and paired run/step/D/r metadata. Hidden views and superseded selections cancel requests; existing bounded-concurrency scheduling is reused. Legacy runs hide the width charts. No adaptive-width controller or telemetry service is added.

## Checkpoints and memory interpretation

Resume uses the existing checkpoint lifecycle:

```bash
python -m run_thog2_owt --resume /path/to/run/ckpt.pt --max-iters 20000
```

The saved geometry and capture configuration are authoritative. Do not resupply selectors or WIDTH/DEPTH options on resume. Width identity includes dimensions, output-by-input shapes, orthonormal-coordinate/stabilization policies, exact versions, fingerprints and independent depth participation. Compatible optimizer state and deterministic probe history are restored. Corrupt basis buffers, incompatible dimensions/bases/geometries and implicit legacy full-width conversion are rejected before tensor loading. Ordinary non-width checkpoint dictionaries retain their original fields. Valid direct Stage6 checkpoints without lifecycle metadata can also recover width configuration.

Dense core matrices contain L(4DA+2DF) scalars; width alone contains L(4rA+2rF), and joint width/depth contains P(4rA+2rF). r=D/2,P=L/2 therefore saves 75% of **core-matrix storage**. This formula is not a total-memory or throughput result. Learned embeddings/head, affine vectors, linear biases, gradients and actual optimizer state are reported separately at their stored precision.

Accounting identifies the one shared fixed basis, dtype conversions, untiled r×D gain-construction temporary, r×r derived G, projected r-bias, and N×r residual activations. Autograd may retain G at multiple sites and checkpoint replays; saved-tensor counters include that storage. Logical saved bytes and distinct nonparameter storage bytes are separately labelled; observed-address reuse can coalesce distinct-storage records. GPU peaks cover a captured completed update including resident parameters/optimizer and basic diagnostics, excluding startup and probes. CPU peak RSS is process-lifetime, includes imports/startup/diagnostics, and cannot be reset to a step peak. Unsupported counters are explicitly unavailable. Basic-capture timing is subtracted from the existing normal-update timing; probes and publishing are reported as instrumentation overhead.

Attention scores, A-wide QKV, KV cache, F-wide MLP hidden activations, vocabulary logits and executed depth lie outside residual-width reduction. The CPU comparison reduces measured learned parameter and optimizer storage, but has worse RSS, slower updates and higher toy held-out loss than dense. See the acceptance record before choosing a width for a larger model.

## Reproduce acceptance

```bash
python -m pytest -q tests/test_residual_width.py tests/test_residual_width_integration.py
python tools/benchmark_residual_width.py --output evidence/residual_width_cpu_comparison.json
python tools/verify_width_disabled_regression.py --baseline /path/to/clean/master --output evidence/residual_width_disabled_regression.json
```

Real-browser acceptance additionally needs Node, Playwright and Chromium. Set `NODE_PATH` to your Node dependency directory and `THOG_WIDTH_BROWSER_EXECUTABLE` to Chromium, then run `python -m pytest -q tests/test_residual_width_browser.py`. Optional `THOG_WIDTH_CHROMIUM_PACKAGE` enables the serverless Chromium package's required launch flags. The browser test starts the HTTP fixture and browser under one parent process. W&B cloud delivery, CUDA training/peaks, multi-rank width training, compiled width execution and meaningful language-model scales remain untested in this CPU environment; the report lists these limits explicitly.
