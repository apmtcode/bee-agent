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
      objective's five pieces — DONE run 9. Pieces 1–4 (capture/schema/dataset/
      replay) existed; **train/infer was entirely missing** (only external mlx/
      axolotl script generation). Run 9 built the model layer.
- [x] Pluggable local-model backend interface for the training runner with a
      deterministic mock backend (so cloud/CI tests pass) and a documented seam
      for a real on-device small model — DONE run 9. `MovementModelBackend` /
      `MovementModel` interfaces + `NgramMovementBackend` (n-gram backoff,
      in-process, serializable). Real on-device model implements the same
      interface.
- [x] Synthetic event-stream generator to validate capture→dataset→replay
      round-trips without real OS input — DONE run 9 (`synthetic-movements.ts`,
      seeded mulberry32).
- [x] Generalization eval harness: measure replay fidelity on held-out but
      related synthetic trajectories — DONE run 9 (`movement-eval.ts`:
      next-token accuracy, LCS replay fidelity, backoff order).
- [ ] Wire `MovementModelBackend` into `LocalTrainingJobStore`/`runner` so a job
      selects `backend: "ngram"` (executes in-process, cloud-testable) vs.
      `backend: "mlx"` (emits external launch script) — one training abstraction,
      mock backend that *actually trains*. Closes objective #2(c/d) end to end.
- [ ] "Motor-babble" curriculum: have the synthetic generator emit noised/
      reordered program variants so the eval harness measures true generalisation
      (novel-but-related movements), not near-memorisation.

## Test-suite health (pre-existing failures found run 9)
Three tests fail on the clean baseline (verified via `git stash`) — they
regressed from the *environment* since run 8's "174/174", not from code. Two
time/timing root causes, both needing a small focused fix:
- [ ] **Injectable clock for remote-heartbeat staleness.** Remote-control tests
      hardcode fixture timestamps; with the clock at 2026-07-18 heartbeats read
      stale → states resolve `quarantined`/`degraded` not `active`
      (`server.test.ts:719`, `app.test.ts:906`). Thread a `now()` clock through
      the staleness computation and have tests inject a fixed clock.
- [ ] **Atomic launch-script state writes.** `renderLaunchScript` in
      `harness/background-tasks.ts` and `training/runner.ts` write state via a
      non-atomic shell `> statepath` (+ python `write_text`) from detached
      processes, racing `readJsonFile` → partial-JSON `SyntaxError`
      (`operator-runtime.test.ts:603`, `app.test.ts:1143`). Write to a temp file
      then `mv` (atomic rename) so readers see old-or-complete, never partial.

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
