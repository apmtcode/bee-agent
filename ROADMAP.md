# bee-agent roadmap

Prioritized backlog for the self-evolution engine. Checked items are done;
unchecked items are queued. Keep this richer than you found it each run.

## Foundations / DX
- [x] Declare build + test tooling in `package.json` and add a `test` script
      (2026-06-22) — nothing could build/test before this.
- [x] Make config loading hermetic in tests via an injectable `configHome`
      (2026-06-22).
- [ ] **Pay down typecheck debt** (surfaced by the `typecheck` script). Full
      `tsc --noEmit` count was **397** on 2026-06-22; now **125**. 🎯 ALL source
      (`src/**` non-test) files typecheck clean since run 7; remaining 125 errors
      are entirely in test files (run 8 cleared 104 by extending the result map).
      Fix per file, no mass-rewrite:
  - [x] `src/capture/` (trajectory-store.ts, replay-service.ts) — DONE run 2.
  - [x] `src/index.ts` (6) — DONE run 3 (barrel alias for cross-module dupes).
  - [x] `src/cli/config.ts` (6) — DONE run 3 (`resolveMergedConfig` helper).
  - [x] `src/control-plane/server.ts` (4) — DONE run 4.
  - [x] `src/orchestrator/operator-runtime.ts` (4 + cascade) — DONE run 4.
  - [x] `src/cli/app.ts` (63 → 0) — DONE run 7. Runs 5–6 built the typed
    `handle()` result map (cron, pairing, monitors, tasks.stop, direct-builder
    sessions); run 7 mapped the composed `sessions.platformControl`
    (`SessionPlatformControlResult`) + `sessions.remoteControl` (refreshed
    status), fixed a real delivery-target bug (`formatDeliveryTargetLabel` covers
    `browser-push`, not just local/webhook), and loosened the status-line
    controller's `stdout` to `NodeJS.WritableStream`.
  - [ ] Map-coverage test: assert every `case "x.y":` in `handle`'s switch has a
    `ControlPlaneResultMap` entry or is explicitly allow-listed as `unknown`, so
    new untyped RPC methods are caught instead of silently `unknown`.
- [ ] Typed client facade `createControlPlaneClient(server)`: one method per
    mapped RPC with inferred params/results, so call sites read
    `client.cronList()` and unmapped methods are a compile error, not `unknown`.
  - [~] Test files: `server.test.ts` (234→118), `app.test.ts` (41→1),
    `gateway-transport.test.ts` (4), `session-stream.test.ts` (1),
    `status-line.test.ts` (1). Most cleared by extending the result map (runs
    5–8). Remaining server.test.ts errors trace to still-unmapped methods —
    `skills.executable.*`, `push.subscriptions.*`, `trajectories.*`, `replays.*`,
    `cron.runs`/misc — plus a few genuine test-only typings. Map the rest, then
    fix residual test-only typings.
- [ ] Add a `verify` npm script (`typecheck && build && test`) and have the
      engine run it as a pre-push self-check each cycle.
- [x] Interim **source-only typecheck gate** — DONE run 7. `tsconfig.src.json`
      (excludes `**/*.test.ts`) + `typecheck:src` script; passes (exit 0). Next:
      have the engine run it as a per-run pre-push self-check.
- [ ] Add a minimal CI workflow mirroring `verify` for human-opened PRs.

## Capability parity (audit reference agents → port gaps)
- [ ] Build a "capability inventory" generator: enumerate bee-agent's exported
      RPC/tool surface (`src/index.ts`) and diff it against `openclaw`,
      `hermes-agent`, `claw-code`, `claude-code` to make parity gaps explicit and
      trackable instead of rediscovered each run.
- [ ] Audit `openclaw` for control-plane/transport features bee-agent lacks.
- [ ] Audit `hermes-agent` (largest reference tree) for orchestration/tooling
      gaps.
- [ ] Audit `claude-code` reference for slash-command / hook coverage gaps.

## 🔴 Top priority (blocker) — restore green test suite
- [ ] **Fix pre-existing background-task recovery failure** (surfaced run 9).
      `npm test` shows 3 failures in `operator-runtime.test.ts`:
      `readJsonFile` throws `SyntaxError: Expected ',' or '}' after property value`
      when `recoverBackgroundTasks` reads a background-task state file. Reproduces
      on base HEAD `3c7b7236` **without** run 9's changes; consistent in isolation.
      Run 8 was 174/174 green at this commit, so the freshly-installed toolchain
      (`@types/node ^26`, `typescript ^6`, `vitest ^4.1.9`) likely exposed a latent
      atomicity race in `writeJsonAtomic`/`readJsonFile` (concurrent read during
      rename on the container overlay fs) or in the reconcile write path. Start by
      dumping the corrupt state file's bytes around offset ~311, then make
      `writeJsonAtomic` fsync+rename fully durable and/or serialize per-file writes.

## Local-movement learning subsystem
Existing scaffolding lives in `src/capture/` (recorder, replay, trajectory,
device/os/browser adapters, consent store, ingestion) and `src/training/`
(exporter, job store/manifest, runner, execution service, **movement-model**).
Next increments:
- [x] Inventory `src/capture` + `src/training` vs the objective's five pieces —
      DONE run 9 (capture/schema/dataset/replay were scaffolded; the missing piece
      was an in-process train→infer→generalize path).
- [x] Pluggable local-model backend interface + deterministic mock backend —
      DONE run 9 (`MovementModelBackend` seam + `NGramMovementBackend` with
      Katz-style context backoff for generalization; `src/training/movement-model.ts`).
- [x] Generalization / replay-fidelity eval harness — DONE run 9
      (`evaluateNextTokenAccuracy`: next-token accuracy + generalized share on
      held-out related trajectories).
- [ ] Synthetic event-stream generator: a helper that emits realistic randomized
      (but seeded/deterministic) `ReplayTimelineEvent[]` streams so capture→dataset
      →train→replay round-trips can be stress-tested at scale without real OS input.
- [ ] Online/streaming backend: `model.observe(sequence)` to update transition
      counts incrementally as reviewed trajectories land (continual adaptation
      between full training jobs).
- [ ] Confidence-gated autonomy: only auto-execute a predicted movement when
      `predictNext` probability + matched-context-length exceed a threshold; else
      defer to the operator. Turns the eval metric into a runtime safety gate.
- [ ] Wire the movement-model into the training runner/job so a job can produce a
      real (non-MLX) policy artifact in the cloud path, and add an RPC to
      predict/rollout from a stored snapshot.

## Innovation backlog
- [ ] Self-check telemetry: each engine run records build/test timing + pass
      counts to a small append-only metrics file to detect regressions in
      project health over time.
- [ ] Coordination guard between the parallel cloud + local self-evolve runs
      (e.g. a lightweight lock/heartbeat file) to avoid duplicated work and
      merge churn.
- [ ] Barrel-collision lint: scan `src/index.ts` re-exports for names exported
      from more than one module and flag them, so duplicate-identifier debt is
      caught at authoring time instead of accumulating silently.
- [ ] Per-module typecheck ratchet: record each module's current `tsc` error
      count to a baseline file and fail if a module regresses above it. Lets the
      engine pay debt down module-by-module without one green-gate blocking
      progress, and prevents backsliding while the total is still > 0.
