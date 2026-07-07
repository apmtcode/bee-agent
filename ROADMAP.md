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
device/os/browser adapters, consent store, ingestion), `src/training/`
(exporter, job store/manifest, runner, execution service — *external* mlx/axolotl
launch plans), and `src/movement/` (run 9: in-process pluggable train/infer +
synthetic corpus + generalization eval). Next increments:
- [x] Inventory `src/capture` + `src/training` vs. the objective's five pieces
      — DONE run 9 (capture/schema/dataset/replay scaffolded; train/infer was
      external-only, so the in-process backend was the gap).
- [x] Pluggable local-model backend interface with a deterministic mock backend
      (so cloud/CI tests pass) + documented seam for a real on-device model —
      DONE run 9 (`MovementModelBackend` + `NgramMovementBackend`, back-off
      n-gram over structural movement tokens; serialize/load round-trip).
- [x] Synthetic event-stream generator to validate the loop without real OS
      input — DONE run 9 (`src/movement/synthetic.ts`, seeded mulberry32,
      train/held-out split with remapped targets).
- [x] Generalization eval harness: replay fidelity on held-out related
      trajectories — DONE run 9 (`evaluateMovementModel`: accuracy, top-K, and
      back-off-only `generalizedAccuracy`).
- [ ] **Behavior-cloning replay driver:** consume a `TrainedMovementModel` and
      drive the (mock) device/browser adapters step-by-step to *re-perform* a
      trajectory — closes record → train → **act** end-to-end with a mock
      actuator so it runs in CI.
- [ ] **Class-backoff token hierarchy** (gesture-kind → gesture-family → any) so
      the model generalizes even when a whole gesture kind is novel; report the
      abstraction level each prediction used.
- [ ] Real on-device small-model backend implementing `MovementModelBackend`
      (behind the same interface the runner's mlx/axolotl plans target).

## Test health
- [ ] **Triage 3 pre-existing suite failures** (surfaced run 9; fail on a fully
      clean tree, so they regressed the run-8 "174/174" note). All in files no
      recent run touched: `app.test.ts` (×2 — `control=active` platform-control
      status; a background-task id assertion), `server.test.ts` (×1 —
      orchestration result shape returning 10 keys where the test expects 2),
      `operator-runtime.test.ts` (×1 — background-task cancel returns
      `undefined`). They involve detached background-task/cron timing + result-map
      shapes; determine flaky-under-parallelism vs. genuine regression, then fix
      so the suite is green and the engine's pre-push gate is trustworthy again.

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
