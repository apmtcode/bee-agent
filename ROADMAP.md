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
(exporter, job store/manifest, runner, execution service, **movement-policy /
mock-policy-backend / movement-eval** — run 9). Next increments:
- [x] Inventory `src/capture` + `src/training` vs. the objective's five pieces
      (run 9): capture ✅, schema ✅, dataset/export ✅, replay ✅, *train-plan* ✅,
      but **infer/repeat/generalize was missing** — now built.
- [x] Pluggable local-model backend interface + deterministic mock backend
      (run 9) — `MovementPolicyBackend`/`MovementPolicyModel` +
      `MockMovementPolicyBackend` (nearest-neighbour repeat + parameter-substitution
      generalize), served via `MovementPolicyEngine`. Documented seam for a real
      on-device small model.
- [x] Synthetic event-stream generator (run 9) — `generateSyntheticMovementFamily`
      (deterministic), plus `trajectoryToMovement` to project real captured spans.
- [x] Generalization eval harness (run 9) — `heldOutGeneralizationCases` +
      `evaluateMovementPolicy` (ordered step fidelity, exact-match rate).
- [ ] Wire a **real on-device small-model backend** behind `MovementPolicyBackend`
      (e.g. an MLX/gguf policy loaded from the runner's artifact) with the mock as
      the deterministic cloud/CI double.
- [ ] Online/incremental policy: append each successful replay as a labelled
      example and re-fit; confidence-gated fallback (below a similarity floor,
      escalate to the LLM planner instead of replaying the nearest match).
- [ ] Connect the policy to the replay engine: turn a `MovementPrediction` into a
      `ReplayTimelineEvent` stream / device-adapter calls so predictions can
      actually drive (simulated) execution end-to-end.

## Reliability / correctness (found while auditing)
- [x] **`shellQuote` POSIX-escaping bug** in `src/harness/background-tasks.ts`
      (run 9): `"'"'"'` → `'"'"'`. A command containing a single quote produced
      an invalid `state.json` that crashed session recovery. Fixed + defensive
      `readState` guard for partial/corrupt state files.
- [ ] **⚠️ TOP PRIORITY — background-task test hermeticity.** `app.test.ts`,
      `server.test.ts`, `operator-runtime.test.ts` each fail one case in *this*
      cloud environment because `startBackgroundTask` spawns a **real detached OS
      process** (default `spawnProcess = spawn`) whose launch script writes real
      state, racing the tests' manual `writeState`. Thread the injectable
      `spawnProcess` / `isProcessRunning` seams (already on
      `BackgroundTaskExecutionService`) up through `StandaloneOperatorRuntime` /
      `OperatorCliApp` / control-plane server and inject no-op mocks in these
      three tests so they are deterministic regardless of host OS. This is what
      currently keeps the full suite from being green.

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
