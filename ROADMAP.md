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

## Local-movement learning subsystem
Existing scaffolding lives in `src/capture/` (recorder, replay, trajectory,
device/os/browser adapters, consent store, ingestion) and `src/training/`
(exporter, job store/manifest, runner, execution service). Next increments:
- [x] Inventory what `src/capture` + `src/training` already implement vs. the
      objective's five pieces — DONE run 9. capture/schema/dataset/replay were
      scaffolded; **train+infer+generalize (2c+2d) were absent in-process** — the
      runner only shelled out to an un-cloud-runnable Apple-Silicon toolchain.
- [x] Pluggable local-model backend interface **+ deterministic in-process model**
      — DONE run 9 (`src/training/movement-model.ts`): `MovementModelBackend`
      interface + `NgramMovementModel` (order-k Markov, Katz backoff, deterministic
      argmax — no RNG/Date, so cloud/CI-safe). Trains on recorded movements,
      repeats the dominant path, and generalizes via suffix backoff. A real
      on-device neural policy can implement the same interface behind the seam.
- [ ] Wire the movement model into the training pipeline as a selectable
      **backend** for `LocalAppleSiliconTrainingRunner` (mode → backend registry),
      so `NgramMovementModel` is the default cloud/CI backend and mlx/axolotl is the
      on-device backend — the runner should delegate command/env/artifact building
      to the chosen backend instead of hardcoding the runtime.
- [ ] Temporal/dwell channel for `NgramMovementModel`: bucket inter-event `ts`
      gaps into duration classes, fold into the token, and have `generate()` emit
      predicted dwell times so replayed movement is human-paced (see run 9 idea).
- [ ] Synthetic event-stream generator to validate capture→dataset→replay
      round-trips without real OS input (now also feeds movement-model training).
- [~] Generalization eval harness — first cut DONE run 9
      (`evaluateSequenceFidelity` + `evaluateNextTokenAccuracy` with backoff count).
      Next: held-out *trajectory-level* replay-fidelity scoring over synthetic
      related trajectories, and a threshold gate the engine can assert on.

## Test reliability (raised priority — blocks the push gate)
- [ ] **De-flake the real-process harness tests.** `app.test.ts`,
      `server.test.ts`, and `operator-runtime.test.ts` each start background tasks
      with the DEFAULT spawn (real OS processes: `tail -f`, etc.) and then assert on
      process liveness / reconciled breaker state. In the cloud sandbox these
      oscillate 3↔4 failures run-to-run (mixed vs degraded breaker state,
      missing-process vs state-running, and — before run 9's `readState`
      hardening — a hard JSON crash on a half-written state file). They pass in the
      canonical env (run 8: 174/174). Fix additively: inject a mock
      `spawnProcess` + `isProcessRunning` into these tests (the seams already
      exist — `FileBackgroundTaskStore(filePath, spawn, isRunning)` and the
      runtime's options) so liveness is deterministic, exactly as
      `background-tasks.test.ts` already does. This is the single biggest thing
      standing between the engine and a fully-green push gate.

## Innovation backlog
- [x] Reliability: `readState` tolerates a half-written state file (run 9) —
      recovery no longer crashes on a mid-write launch-script state file.
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
