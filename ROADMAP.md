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
- [ ] Inventory what `src/capture` + `src/training` already implement vs. the
      objective's five pieces (capture → schema → dataset → replay → train/infer)
      and write the gap list here before adding code.
- [x] Pluggable local-model backend interface for the training runner with a
      deterministic mock backend (so cloud/CI tests pass) and a documented seam
      for a real on-device small model — DONE run 9. `MovementTrainingBackend`
      interface + `MockMarkovMovementBackend` (backoff n-gram) in
      `src/training/{movement-model,mock-movement-backend}.ts`.
- [x] Synthetic event-stream generator to validate capture→dataset→replay
      round-trips without real OS input — DONE run 9 (`synthesizeMovementDataset`,
      deterministic seeded grammar walk).
- [x] Generalization eval harness: measure replay fidelity on held-out but
      related synthetic trajectories — DONE run 9 (`evaluateReplayFidelity` +
      `generateMovementSequence`; generalization test trains on 40 workflows,
      evals unseen trajectory via backoff).
- [ ] Real on-device backend behind `MovementTrainingBackend`: an
      `OnnxMovementBackend`/`TorchMovementBackend` over the token vocabulary
      (small GRU/transformer), with the mock as the CI fallback and a `backendId`
      on the training job manifest so a job selects its backend. Then the runner's
      shell-script path and the in-process path become two backends behind one
      interface.
- [ ] Wire the movement backend into the training runner/job flow: let a
      `LocalTrainingJobManifest` train in-process via the mock backend (producing
      a persisted model artifact + replay-fidelity eval) as an alternative to the
      Apple-Silicon shell-script path — closes the objective-2 loop end to end.

## Known bugs (discovered, not yet fixed)
- [ ] **Fragile bash JSON state writer** in `src/harness/background-tasks.ts`
      `renderLaunchScript` (and the identical pattern in `src/training/runner.ts`):
      the initial "running" state is produced by `sed` surgery on a JSON payload
      inside a double-quoted shell string (`s/"$$"/$$/g` with `$$`
      shell-expanded on the LHS; embedded command quotes/newlines corrupt the
      payload). This writes **invalid JSON** on non-GNU-coreutils environments
      (repro: this cloud sandbox — 3 suite tests fail with `SyntaxError` in
      `readJsonFile`; GNU machines mask it, hence earlier "174/174"). **Fix:**
      write the running state via a `python3` heredoc (json handles all escaping),
      mirroring the existing completion writer — verified to fix 2 of 3 tests
      locally in run 9. **Blocked on:** the 3rd test (`server.test.ts` breaker) is
      *racy* — it launches 3 real detached `sleep 5` tasks with
      `isProcessRunning:()=>false` and asserts exactly 2 are "running" at
      recovery; a faster/correct state writer wins the write race for the 3rd
      task (failureCount 3/threshold 3/"paused" vs expected 2/"degraded"). Do the
      writer fix **and** de-race that test (e.g. write initial state
      synchronously in `execution-service.launch()`, or make the test control the
      3rd task's state deterministically) as one focused unit.

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
