# Stage 3E checkpoint permanent removal

Implemented on `migration/stage3e-rubbish`, 5 November 2026.

The model-owned management service adds explicit `purge` capability. Recovery
and immutable descriptor protocol remain version 1. Existing keep-all/latest/N
workers still perform only recoverable removal; cached policy never authorises
an irreversible effect. INSTRA's fixed policy authority admits automatic expiry
one exact operation/episode at a time.

Purge requires a removed whole bundle, confirmed inactivity, known ownership,
unchanged registered members and absence of all independent protections. It
persists intent before unlink, journals each member, retains partial-purge
unavailability, and retries the original scope. Source containment and no-follow
parent descriptors prevent redirecting an unlink through a replaced path.

Only matching registered aliases are removed. An alias repointed at a surviving
checkpoint stays intact. External payloads are not unlinked; only their managed
registration is removed. Confirmed unlinked bytes do not claim filesystem space
freed. A minimal permanent tombstone prevents old files being re-published under
the purged bundle identity. Child recovery from its own checkpoint is independent
of a permissibly purged Parent payload.

Upgrade both model repositories and the standalone application on every participating
Host before enabling E mutations or expiry; then restart INSTRA so Agent runtime
fingerprints are refreshed. Do not downgrade to pre-E writers against affected
stores. Preserve checkpoint management journals and pins during rollback.

Verification is recorded in standalone `docs/STAGE3E_ACCEPTANCE.md`. CPU recovery
checks include physically purging a Parent before resuming its Child. GPU,
independent physical Hosts and full Stage 3 endurance acceptance remain pending.
