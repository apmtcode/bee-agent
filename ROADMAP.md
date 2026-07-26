# bee-agent roadmap

Prioritized backlog for the self-evolution engine. Checked items are done;
unchecked items are queued. Keep this richer than you found it each run.

## Foundations / DX
- [x] Declare build + test tooling in `package.json` and add a `test` script
      (2026-06-22) — nothing could build/test before this.
- [x] Make config loading hermetic in tests via an injectable `configHome`
      (2026-06-22).
- [x] **Eliminate background-task test flakiness** (2026-07-25, run 9) — the
      suite was non-deterministic (1–4/174 random failures) because
      `startBackgroundTask` spawned real subprocesses whose async state/output
      writes raced assertions. Threaded `backgroundTaskSpawnProcess` /
      `backgroundTaskIsProcessRunning` through `OperatorCliAppOptions` and
      injected no-op spawn stubs into all background-task tests. Now 10/10
      consecutive green. Restores the engine's verification gate.
- [ ] **Flake sentinel in the pre-push self-check**: run `vitest run` twice each
      cycle and only treat the suite as green if *both* pass — a single lucky run
      must not gate a push (the exact failure mode run 9 fixed).
- [ ] **Test-spawn lint**: flag `startBackgroundTask` / real `spawn` in `*.test.ts`
      that don't inject a spawn stub, so subprocess-race flakiness can't return.
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
(exporter, job store/manifest, runner, execution service, **movement-model**).
- [x] Inventory `src/capture` + `src/training` vs. the objective's five pieces
      (2026-07-26, run 10). Result: **capture, schema, dataset, replay all exist**;
      the gap was an in-process **train→infer** model for pieces (c) repeat and
      (d) generalize.
- [x] **Pluggable local-model backend + deterministic mock** (2026-07-26, run 10).
      `src/training/movement-model.ts`: `MovementModelBackend` interface (the seam
      for a real on-device model) + `NgramMovementBackend` mock (Markov n-gram,
      specific/generic two-level index → exact repeat + tool-level generalization,
      fully deterministic). `rolloutMovement` replay driver + trajectory/replay
      adapters. 12 tests.
- [ ] **Generalization eval harness** (now cheap to build on the n-gram backend):
      split synthetic trajectory *families* into train/held-out, roll the policy
      out from held-out seeds, score predicted-vs-ground-truth action fidelity as a
      single `generalizationScore ∈ [0,1]`. Makes "does it generalize?" a tracked
      regression metric and lets mock vs. future on-device backends be compared on
      equal footing.
- [ ] **Synthetic event-stream generator**: parametric generator of trajectory
      families (shared pattern, varied instances) to feed both the model tests and
      the eval harness without real OS input. Extract the `fileEditSequence`-style
      helpers from `movement-model.test.ts` into a reusable `src/training/synthetic`
      module.
- [ ] Wire a trained `MovementPolicy` into the runtime as an **action-suggestion
      provider** — surface predict-next-action (with confidence + `source`) to the
      operator so learned movements assist live sessions, not just offline replay.
- [ ] Second concrete backend behind `MovementModelBackend` (e.g. a
      frequency-smoothed or embedding-nearest-neighbour policy) to prove the
      interface is genuinely pluggable and give the eval harness a comparison point.

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
