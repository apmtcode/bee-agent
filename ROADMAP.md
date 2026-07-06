# bee-agent roadmap

Prioritized backlog for the self-evolution engine. Checked items are done;
unchecked items are queued. Keep this richer than you found it each run.

## Foundations / DX
- [ ] **🔴 TOP PRIORITY — fix pre-existing flaky integration failure.** 3 tests
      (`operator-runtime`, `cli/app`, `control-plane/server`) fail
      non-deterministically (run 8 logged 174/174; run 9's base shows 3 red).
      Diagnosed run 9: `operator-runtime.test.ts` starts a **real `tail -f`
      subprocess** whose bash launch script writes single-line state JSON via
      printf/sed, and `recoverBackgroundTasks` reads it **mid-write** → JSON
      `SyntaxError` at ~col 312. Fix options: (a) have the recovery read path
      (`readJsonFile`/`BackgroundTaskExecutionService.readState`) tolerate a
      transient parse failure with a single short retry (careful: don't mask real
      corruption); (b) make the test not spawn a live long-running process, or
      settle it before recovery; (c) write subprocess state atomically like
      `writeJsonAtomic` instead of printf/sed. Prefer (c)+(b).
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
      for a real on-device small model. **DONE run 9** — `MovementModelBackend`
      interface + `MarkovMovementBackend` (order-k Markov + stupid-backoff) in
      `src/training/movement-model.ts`. Reproduces recorded sequences (obj c),
      generalizes via backoff (obj d), deterministic, `toJSON`/`fromJSON`.
- [x] Synthetic event-stream generator to validate capture→dataset→replay
      round-trips without real OS input. **DONE run 9** —
      `synthesizeMovementSequences` (deterministic, seed-offset disjoint splits).
- [x] Generalization eval harness: measure replay fidelity on held-out but
      related synthetic trajectories. **DONE run 9** — `evaluateMovementModel`
      (top-1 accuracy, generalization/backoff rate, mean confidence).
- [ ] Wire the movement model into the runner/execution-service: pick backend by
      `RlTrainingConfig`/`SftTrainingConfig`, persist the trained model (`toJSON`)
      as a job artifact, and expose train/infer over the control-plane RPC surface.
- [ ] Replay-fidelity reward for the RL runner: reuse `evaluateMovementModel` as a
      deterministic in-process reward so `reward-model: replay-manifest` has a
      concrete implementation, not just an external hook.
- [ ] Richer tokenization: keep an argument "slot" alongside the structural verb
      token so generalization can be measured *with* and *without* literals, and so
      a real backend can condition on both.

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
