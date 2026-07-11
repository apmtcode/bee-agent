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
(exporter, job store/manifest, runner, execution service, **movement-model**,
**synthetic-stream**). Next increments:
- [x] Inventory what `src/capture` + `src/training` already implement vs. the
      objective's five pieces (run 9): capture/schema/dataset/replay existed; the
      runner only shelled out to external Python — **no in-process train/infer**.
- [x] Pluggable local-model backend interface for the training runner with a
      deterministic mock backend (run 9): `MovementModelBackend` +
      `MarkovMovementBackend` (order-k Markov w/ backoff, serializable), plus a
      documented seam for a real on-device small model.
- [x] Synthetic event-stream generator to validate capture→dataset→replay
      round-trips without real OS input (run 9): `generateSyntheticMovementDataset`
      (seeded mulberry32) + `splitMovementDataset`.
- [x] Generalization eval harness (run 9): `evaluateMovementModel` — next-step
      top-1 accuracy + coverage on held-out related synthetic trajectories.
- [ ] Wire the movement model into the training runner/execution path: let
      `MarkovMovementBackend` produce a persisted `model.json` artifact from a
      reviewed export's replay manifests, alongside the existing mlx/axolotl
      launch plan, so a real train step exists even without Apple-silicon tooling.
- [ ] Pluggable tokenizer for `MarkovMovementBackend` (structural
      `tool:gesture` ↔ instance `tool:gesture:target`) + a reward-weighted
      variant that biases next-step probabilities by trajectory `outcome.reward`.
- [ ] Round-trip a *device-adapter* capture stream (gestures) through
      `movementSequenceFromTrajectory` → train → `generate` and assert replay
      fidelity, closing the capture→train→replay loop end-to-end.

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
- [ ] **De-flake `server.test.ts`'s big "handles session…" test** (found run 9):
      its breaker section races — a platform breaker auto-trips to `degraded`
      before a `mixed` assertion depending on wall-clock timing (flaky even
      single-threaded). Fix by injecting a deterministic clock and/or disabling
      background reconcile in the test harness. This is the last pre-existing
      full-suite flake in this environment now that the state-writer sed bug is
      fixed.
- [x] **Make the launch-script state writer portable** (run 9): replaced the
      fragile `printf|sed` init-`state.json` write (corrupted escaped JSON on GNU
      sed 4.9) with a Python `json.dumps` writer in `background-tasks.ts` and
      `training/runner.ts`.
