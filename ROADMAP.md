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
      objective's five pieces (2026-07-23, run 9). Gap found: capture→schema→
      dataset→replay existed, but **train/infer only shelled out to python** —
      nothing ran in-process, so parts (c)/(d) were unvalidatable in the cloud.
- [x] Pluggable local-model backend interface for movement learning with a
      deterministic in-process backend (2026-07-23, run 9) — `src/movement/`:
      `MovementModelBackend` seam + `MarkovMovementBackend` (train/predict/
      generate, JSON artifact). Real on-device small model plugs into the same
      interface; the python runner becomes one opt-in backend behind it.
- [x] Synthetic event-stream generator (2026-07-23, run 9) — seeded, template-
      driven `generateSyntheticDataset` validates dataset→train→replay round
      trips without real OS input.
- [x] Generalization eval harness (2026-07-23, run 9) — `evaluateNextEventAccuracy`
      (top-1/top-k/backoff) + `evaluateReplayFidelity` on held-out sequences.
- [ ] `MovementTrainingBridge`: pull reviewed trajectories from
      `FileTrajectoryStore` → `movementSequenceFromTrajectory` → `MovementDataset`
      → train a `MovementModelBackend`, and persist the artifact alongside the
      existing `LocalTrainingJobManifest`. Unifies the in-process (cloud-testable
      default) and python (real on-device) training paths behind one seam.
- [ ] Sampling decoder for `generate()`: temperature + top-p with an n-back
      repetition penalty, so replay reproduces internally-repeated-token
      workflows (a pure argmax Markov rollout self-loops on e.g. consecutive
      `scroll` events) and can explore plausible related variations.
- [ ] Wire the movement backend into the control-plane RPC surface + CLI so the
      operator can train/eval a movement model and inspect fidelity metrics.

## Innovation backlog
- [ ] **Make background-task process-liveness injectable** (test hermeticity).
      3 tests fail in the cloud sandbox — `operator-runtime.test.ts`,
      `control-plane/server.test.ts`, `cli/app.test.ts` (all the recover/sync/
      remote-repair path) — because the runtime does a real pid-liveness check
      that reports `missing-process` when the recorded pids aren't alive here.
      Inject the liveness probe (like `SpawnProcess` already is) so tests are
      deterministic regardless of the host's process table. Noted run 9.
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
