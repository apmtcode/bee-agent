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
- [x] Inventory `src/capture` + `src/training` vs. the objective's five pieces
      (2026-07-07, run 9). Gap found: capture→schema→dataset→replay existed, but
      train/infer stopped at *plan generation* (`runner.ts` emits shell scripts
      for on-device MLX/axolotl) — no in-process learnable model.
- [x] Pluggable local-model backend interface for the training runner with a
      deterministic backend (so cloud/CI tests pass) and a documented seam for a
      real on-device small model — DONE run 9 (`src/training/movement-model.ts`:
      `MovementModelBackend`/`MovementPolicy`/`MovementModelArtifact` +
      `DeterministicNGramBackend`).
- [x] Synthetic event-stream generator to validate capture→dataset→replay
      round-trips without real OS input — DONE run 9
      (`generateSyntheticMovementTrajectories`, seeded LCG + workflow grammar).
- [x] Generalization eval harness: measure replay fidelity on held-out but
      related synthetic trajectories — DONE run 9 (`evaluateMovementPolicy`,
      splits accuracy into exact-memory vs generalized/backoff hits).
- [ ] Imitation-vs-generalization scorecard: each engine run holds out 20% of
      reviewed trajectories, trains `DeterministicNGramBackend`, and appends
      `{exactMemoryHits, generalizedHits, accuracy}` to a metrics file — the
      first quantitative self-evolution KPI for objective #2d.
- [ ] Config-driven backend selection (`movement.backend: "deterministic-ngram"
      | "mlx-local"`) so the runner picks the in-process learner for eval and the
      on-device backend for real training behind one `MovementModelBackend`.
- [ ] Wire the movement model into the training runner/execution-service so a
      reviewed export can be trained + evaluated in-process (cloud path) in
      addition to the shell-plan (on-device) path.

## Test health (discovered run 9)
- [ ] **Fix 3 pre-existing, time/environment-dependent test failures** (surfaced
      2026-07-07; repo unchanged since run 8's 174/174, so wall-clock time is the
      only changed variable). Focused fix = inject/freeze a clock (and a
      pid-liveness seam) in these tests, no source behaviour change:
  - `operator-runtime.test.ts:605` — `recoverBackgroundTasks` depends on whether
    a hard-coded pid (`5678`) is alive in the container → flaky.
  - `server.test.ts:719` + `app.test.ts:906/1098` — gateway heartbeat
    staleness/quarantine asserted against `Date.now()` with stale Jan-2026
    fixtures.

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
