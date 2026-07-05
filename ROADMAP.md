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
(exporter, job store/manifest, runner, execution service). **Run 9 added the
in-process model loop in `src/movement/`** — the piece that actually *learns*
from a dataset and predicts/generalizes, complementing the external-Python
(mlx/axolotl) delegation path in `src/training/runner.ts`.
- [ ] Inventory what `src/capture` + `src/training` already implement vs. the
      objective's five pieces (capture → schema → dataset → replay → train/infer)
      and write the gap list here before adding code.
- [x] Pluggable local-model backend interface (`MovementModelBackend`) with a
      deterministic in-process backend (`MarkovMovementBackend`, variable-order
      Markov + stupid-backoff) so cloud/CI tests pass — DONE run 9. The real
      on-device small model plugs in behind the same seam.
- [x] Synthetic event-stream generator (`generateSyntheticMovementDataset`, seeded
      mulberry32 PRNG + task grammars) to validate the tokenize→train→generate
      round-trip without real OS input — DONE run 9.
- [x] Generalization eval harness (`evaluateMovementModel` + `splitMovementDataset`)
      measuring next-movement top-1/recall and backoff-order distribution on
      held-out related trajectories — DONE run 9 (held-out top-1 ~0.46, recall
      ~0.80, mean backoff order < trained order ⇒ real generalization).
- [ ] Trajectory→movement bridge in the control plane: expose `trajectories.*`
      reviewed spans → `tokenizeTrajectoryDataset` → train → `movement.predict`
      RPC so the CLI can drive the model. (`tokenizeTrajectorySpan` already maps
      the real `TrajectorySpan`/gesture schema to `MovementSequence`.)
- [ ] Model persistence: wire `MarkovModelState` serialize/restore through a
      file store (mirror `FileTrainingJobStore`) so trained movement models
      survive process restarts.
- [ ] Second backend behind the seam: a positional/attention or
      frequency-smoothed backend to compare generalization vs. the Markov
      baseline on the same eval harness (the harness is backend-agnostic).

## Known blockers (triage first)
- [ ] **3 pre-existing test-file failures** discovered run 9, present on branch
      HEAD *before* run 9's additive change (verified by stashing):
      `operator-runtime.test.ts` ("starts, syncs, recovers…"),
      `app.test.ts` (background/monitor/cron + session-lifecycle), and
      `server.test.ts` ("session, transcript, approval, trajectory…"). Root cause:
      these tests spawn **real subprocesses** via `renderLaunchScript`
      (`src/harness/background-tasks.ts:732`, `bash`/`python3`/`date` + a
      `printf | sed > state.json`) whose async single-line state write races /
      corrupts the JSON the test later reads (`SyntaxError … position 311`).
      Deterministic in this sandbox. Fix idea: make `startBackgroundTask`'s state
      init happen in-process (or gate the launch-script write behind a test seam)
      instead of a raced subprocess, so `readState` never sees a half-written /
      sed-mangled file. Out of scope for run 9's movement increment.

## Innovation backlog
- [ ] Self-check telemetry: each engine run records build/test timing + pass
      counts to a small append-only metrics file to detect regressions in
      project health over time.
- [ ] Online/continual movement learning: incrementally update the Markov
      counts as new reviewed trajectories arrive (append-only count merge)
      instead of retraining from scratch — a natural fit for the hourly engine.
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
