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
      objective's five pieces — DONE run 9. Had: capture (adapters→recorder),
      schema (trajectory/replay), dataset (exporter→ReviewedExportManifest),
      replay (ReplayManifest). Missing: an in-process trainable/inferable model.
- [x] Pluggable local-model backend interface for the training runner with a
      deterministic mock backend and a documented seam for a real on-device
      model — DONE run 9. `MovementModelBackend` interface +
      `MarkovMovementBackend` (deterministic back-off n-gram) +
      `createMovementBackend()` registry seam (`src/training/movement-model.ts`,
      `markov-backend.ts`). 18 tests.
- [~] Synthetic event-stream generator to validate capture→dataset→replay
      round-trips without real OS input. Partial: run 9's tests build datasets
      from synthetic trajectories/replays and round-trip train→generate→score.
      Next: a reusable generator that emits parameterized device/os/browser
      event streams (varying apps, gestures, lengths) for larger evals.
- [x] Generalization eval harness — DONE run 9.
      `MovementTrainingService` + `evaluateMovementModel` (teacher-forced token
      accuracy, greedy exact-match replay fidelity, mean log-prob) with a
      deterministic held-out split (`src/training/movement-training-service.ts`).
- [ ] **Intent-conditioned movement policy** (run 9 idea): prefix each sequence
      with a coarse goal token (from trajectory outcome / skill title) so
      `generate({ seed: [START, goal] })` yields the movement path *for that
      goal* — turns the Markov replayer into a small goal→movement policy and is
      the bridge to #2d generalization across related goals.
- [ ] `MovementReplayEngine`: map generated movement tokens back to concrete
      device/os/browser adapter calls so a trained model can drive on-device
      execution (guarded behind the simulated adapter for cloud tests).

## Background-task reliability (NEW — pre-existing flaky suite, run 9)
The full `npm test` is **flaky-red**: 3 "big lifecycle" tests
(`orchestrator/operator-runtime.test.ts`, `control-plane/server.test.ts`,
`cli/app.test.ts`) fail non-deterministically under parallel load on the
contended cloud host (1–4 failures/run). They pre-date run 9 (bee-agent HEAD
unchanged since run 8; only host load differs). This blocks the pre-push green
gate and forces work onto the dev branch. Fix in a **dedicated focused run**
(prototyped + reverted in run 9 to keep the movement diff clean):
- [ ] Fix `renderLaunchScript` (`src/harness/background-tasks.ts`): it hand-rolls
      the running-state JSON via `printf '%s' … | sed`, which yields **invalid
      JSON** for commands containing single quotes/newlines, and its pid
      substitution `s/"$$"/…/` never matches (`$` is a sed anchor). Replace with
      a **quoted heredoc** (`cat > state <<'X'` — zero shell interpretation of
      the `JSON.stringify` output) + a tiny python step to fill pid/timestamps
      (reuses the existing safe python state-writer pattern). The same
      `printf | sed` bug also exists in `src/training/runner.ts` — fix both.
- [ ] Fix the `bash -lc <shellQuote(command)>` path that mangles special-char
      commands (a command with quotes/newlines exits 2 after the JSON fix).
- [ ] Make the 3 failing tests **deterministic** by injecting the codebase's
      existing **mock spawn** (`backgroundTaskSpawnProcess: () => ({ pid, unref(){} })`,
      as `background-tasks.test.ts` already does) so launch scripts never execute
      real subprocesses that race the tests' explicit `writeState`. `OperatorCliApp`
      needs a small additive `backgroundTaskSpawnProcess?`/`…IsProcessRunning?`
      option (mirroring the existing `configHome` test-seam) threaded to its
      runtime, so `app.test.ts` can inject too.
- [ ] Then promote the run-9 movement subsystem from the dev branch to `main`.

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
