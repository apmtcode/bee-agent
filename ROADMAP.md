# bee-agent roadmap

Prioritized backlog for the self-evolution engine. Checked items are done;
unchecked items are queued. Keep this richer than you found it each run.

## 🔴 Regressions to fix first (found run 9)
Run 8 logged a green suite (174/174); on the clean committed tree the suite is now
**171/174**. Three pre-existing failures, unrelated to run 9's additive change:
- [ ] `src/orchestrator/operator-runtime.test.ts` — `readState` throws
      `SyntaxError: Expected ',' or '}' ... at position 311` from a background-task
      state file (`FileBackgroundTaskStore.reconcileTask` → `readState`). Likely a
      malformed/partially-templated state JSON (see `$$`/`__OPENCLAW_STARTED_AT__`
      substitution paths). Highest priority — a real parse bug, not a test typing.
- [ ] `src/cli/app.test.ts` — expects `control=active` but gets
      `control=degraded:automatic retryable failures 2/2 background task
      missing-process` (probably a cascade of the same background-task issue).
- [ ] `src/control-plane/server.test.ts` — orchestration method result shape:
      `expected {…10} to match {…2}` (43 props). Assert the mapped result rather
      than the whole object, or fix the handler's returned shape.

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
      for a real on-device small model. — DONE run 9 (`src/training/model-backend.ts`:
      `MovementModelBackend` interface + `MovementModelBackendRegistry` +
      `MarkovMovementBackend` variable-order backoff model; train/predict/generate/
      serialize; 11 tests).
- [ ] Synthetic event-stream generator to validate capture→dataset→replay
      round-trips without real OS input. (Partial: run 9's tests hand-build
      `ReplayTimelineEvent` streams and tokenize them; still want a reusable
      generator that emits varied mouse/keyboard/window streams programmatically.)
- [x] Generalization eval harness: measure replay fidelity on held-out but
      related synthetic trajectories. — DONE run 9 (`evaluateReplayFidelity`:
      prompt→roll-forward, longest-common-prefix fidelity in [0,1]; tested on
      exact + held-out-divergent sequences).
- [ ] Behavior-cloning backend (tiny in-process softmax next-token classifier)
      as a second backend to A/B against Markov backoff via the same fidelity
      harness — proves the backend interface is genuinely model-agnostic (run 9 idea).
- [ ] Wire the trained movement model into `LocalTrainingExecutionService` /
      `LocalAppleSiliconTrainingRunner` so a reviewed export can be
      round-tripped through the in-process backend as a dry-run before emitting
      the real MLX/axolotl launch script.

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
