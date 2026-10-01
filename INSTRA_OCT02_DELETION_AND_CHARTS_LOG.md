# Instra: distributed deletion, wall clocks and curve controls

Branch: `distributed`. Validation completed 2026-10-01 UTC.

## Delivered behavior

- Producer deletion hides chart runs immediately, retains their chart databases
  during confirmation, and publishes one ordinary JSON notice per run. Expected
  replies snapshot the producer's existing host list, excluding itself.
- Monitoring handles notices before chart acquisition and sends ordinary JSON
  receipts over its existing SSH connection. Cached-copy removal and no-op
  removal send the same successful receipt. Failed notice transfers pause that
  producer's acquisition; failed receipt uploads retry from durable local files.
- Notices, expected hosts, received replies and original request time survive
  restarts. Missing notice publications are recovered. Duplicate requests retain
  their original identity, timeout and expected host set. Receipt matching checks
  producer, run, source path, request and responding host.
- Final chart cleanup follows all confirmations or expiry. Event logs distinguish
  confirmation from timeout, and Settings displays outstanding host names.
  `Deletion confirmation timeout` defaults to seven days, is global and saved,
  and each request captures its configured value.
- Local force deletion of acquired runs requires the specified confirmation.
  It removes the local copy, acquisition record and chart caches in the background,
  without authoritative deletion or durable suppression. The next successful
  ordinary monitoring cycle reacquires a still-existing producer run.
- Acquisitions, receipt processing and force deletion are serialized per producer.
  A force request arriving during acquisition also wins before final publication.
  Failed cleanup cannot issue a successful deletion receipt. Cached files shared
  with an unaffected acquired run remain available to that run.
- Grid deletion keeps producer authority, durable Grid identity and active-Grid
  guards. The confirmation includes its run count. Delete menu sections have
  vertical separation. Original producer checkpoints, training logs and W&B data
  remain outside deletion scope.
- Eye-open rows have slightly lighter text than the mouse-selected row. Current
  front curves are thicker and their run rows visibly highlighted. `z+` and `z-`
  cycle in opposite directions; `!+` and `!-` retain and remove independent bolding.
  Front-row identity participates in the render-cache signature so changing curve
  order updates the table reliably without continuous unnecessary row rebuilding.
- Progress and History show aligned run start/end wall clocks. History also shows
  Grid start/end before T. Full dates are available in tooltips. Unknown legacy
  clocks and unfinished/retried ends remain blank. Recoverable clocks are stored
  and backfilled into the existing durable JSON Runner history used by the duration
  estimator; no new database engine is introduced.
- Long-lived W&B scanners and chart readers have bounded caches. Deleting a run
  clears those caches; reacquisition creates fresh readers. Manifest disk flushes
  no longer hold the lock used by catalogue reads. Runtime assets are versioned.

## Validation

- All relevant non-browser Instra backend tests, local chart dashboard/lifecycle,
  time-axis and processing pairing regressions passed together: 239 tests and
  32 subtests. The older September 1 CLI module passed separately: three tests
  and six subtests. Its inherited global argparse import-order conflict remains
  when combined with model imports; the training CLI was not changed.
- All 42 non-browser JavaScript regression scripts passed.
- Chromium and Firefox both passed the new rendered deletion/menu, eye/front,
  curve cycling/bolding/polling and aligned wall-clock suite, and the existing
  complete History/download/header/race/colour/visibility suite. Chromium also
  passed host-isolated Grid eyes, local deletion eligibility, dense field outlines,
  Ready status, header sizing and actual run-name hover coverage. Screenshots of
  the timing tables and controls were inspected.
- New backend cases cover single and bulk deletion, full/partial receipt sets,
  timeout capture/expiry, restart recovery, offline backlog, no-copy processing,
  failed transfers/uploads/cleanup, duplicates, mismatched receipts, path guards,
  acquisition races, background force deletion and reacquisition, disabled-host
  cleanup, settings persistence and shared cached logs.
- The older lifecycle test's mocked clock had only two readings although store
  construction and final processing capture also read it. The fixture now supplies
  all four readings; production chart-store behavior is unchanged.
- Changed Python and JavaScript files compile, and `git diff --check` passes.

## Practical limits and rollout

The live Scruffy/Dreedle stall was not reproduced here: this workspace cannot
observe their running backends or perform an hours-long two-host soak. This update
addresses verified cache growth and blocking/repeated work. Existing `/api/health`
and bounded `backend-stalls.log` thread dumps remain available if a stall recurs.

The older Selenium launch harness cannot run here because its launcher requires a
Unix Node Agent socket unavailable in this environment. Browser validation instead
used the dashboard fixture in real Chromium and Firefox. GPU training was not
altered or exercised as part of this update.

Expected confirmations intentionally use only the producer's existing host list.
Unlisted monitors are not awaited; listed hosts that never monitor may remain
outstanding until expiry. A host returning after notice expiry may use manual
force deletion. There is no monitoring registration, Node Agent RPC, new network
protocol, server or SQLite extension.

Update the `distributed` checkout and restart Instra on both producing and
monitoring hosts, then hard-refresh the browser. Every monitoring host must receive
the update to process notices and upload receipts.

```bash
cd ~/git/thog2
git fetch origin
git switch distributed
git pull --ff-only origin distributed
git log -1 --oneline
```
