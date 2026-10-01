# Instra: local GPU queues and long-session responsiveness

Date: 2026-10-01. Branch: `distributed`.

## Result

- New Grids use the owner's permanent host prefix, for example `SCR-00021`
  and `DRE-00021`. A colliding host receives three random uppercase letters.
  Prefixes survive restarts, display-name changes and host removal/re-addition.
  Existing `G-` artifacts and history retain their names.
- Eye selection, Grid colours and embedded Multiview use the durable Grid UUID.
  Older records without a UUID use the owner/producer identity plus the tag.
  This keeps genuine distributed Grid members together and separates unrelated
  same-number Grids. Embedded visibility settings also use this identity.
- Saving a Recipe enters `ready`. Preview does not reserve a GPU or enter a
  reservation queue. Launch revalidates and creates queued work. Saving remains
  possible while no execution host or GPU is available.
- Each Node Agent persists its own per-GPU waiting queue and exclusive owner.
  The first eligible waiter can acquire a free GPU. An owner keeps the GPU
  between compatible runs, then releases that GPU independently when it has no
  compatible pending work. Selected busy GPUs can join later. Each Grid's
  Instra controller assigns distinct pending run identities across hosts.
  Grid dispatch does not require the legacy shared Runner Master.
- Tight Repeat preserves fixed placement. Converting remaining work to Loose
  enables dynamic placement while retaining running and completed attempts.
  Grid history cannot be removed while GPU release is still pending.
- Grid-wide chart deletion uses one catalogue pass and cached canonical
  lookups. Remote copies are excluded, foreign Grid UUIDs are rejected, and
  active Grids must be stopped first. Existing remote deletion guards remain.
- Loss hover text carries the full run name independently of the legend.
  `GB Pk` has sufficient column width. Dense-only Recipes outline incompatible
  supplied fields in red; mixed dense/depth Grids strip incompatible options
  from dense references before deduplication.
- Runtime asset URLs are versioned so an update loads the revised controls.

## Hang mitigation and evidence

Slow host reservation, launch, stop, release and log calls no longer hold the
controller lock. Decisions are persisted before remote mutation; concurrent
Save/Stop changes cause stale worker results to be discarded and reconciled.
Node Agent status and discovery also bypass slow mutation/preflight locks.

W&B metric history retained in memory is capped at 12,800 points per series.
Compaction preserves historical extrema/endpoints and the most recent samples;
the original history files remain intact. Plot responses retain their existing
1,600-point limit. Canonical run lookup avoids repeatedly walking all run
directories during polling and batch deletion.

`/api/health` reports uptime and active request paths without waiting for
controller/catalogue locks. A request lasting over 30 seconds triggers an
all-thread stack dump in `backend-stalls.log` in the existing Instra state
directory (with bounded rotation). Request queries and bodies are omitted.

The hours-long hang was not reproduced on Scruffy: this environment has no
connection to its live backend or CUDA GPUs. These changes address verified
unbounded retention and blocking paths; a live-host soak is still needed to
confirm the reported hang is resolved. The diagnostics provide evidence if it
recurs.

## Validation

- 166 Python tests plus ten subtests passed together, covering Runner,
  Networks, Monitoring, deletion, diagnostics, long histories and dashboard
  compatibility.
- The older September 1 CLI compatibility module passed separately: three
  tests plus six subtests. Combining it with model imports exposes its existing
  global argparse patch/import-order conflict; no CLI implementation was changed
  for that issue.
- All 39 non-browser JavaScript regression scripts passed.
- Chromium passed the existing full dashboard History/download/visibility
  suite and the new host identity/dense validation/Ready/header/actual hover
  suite. The screenshots were inspected.
- New scheduling cases cover busy GPU late entry, shared pending work,
  independent release, FIFO eligibility, cancellation without foreign release,
  cross-host unique assignment, controller restart and concurrent Save/Stop.
- A 100,000-sample regression verifies bounded metric retention, recent detail
  and preservation of a historical spike. Changed Python/JavaScript files
  compile; `git diff --check` passes.

## Update

Update the `distributed` checkout on both Scruffy and Dreedle, then restart each
Instra backend using its usual launcher. The launcher replaces a verified older
local Node Agent with protocol 3; accepted training children and persisted
attempt identities remain available for reconciliation. All connected execution
hosts need this version before using local GPU queues.

Existing launched Grids keep their prior scheduling behavior. Newly launched
Grids and explicit Tight-to-Loose conversions use the new dynamic queues.
Existing logs, checkpoints, saved Recipes and `G-` Grid history remain readable.
