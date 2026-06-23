# bee-agent roadmap

Prioritized backlog for the self-evolution engine. Checked items are done;
unchecked items are queued. Keep this richer than you found it each run.

## Foundations / DX
- [x] Declare build + test tooling in `package.json` and add a `test` script
      (2026-06-22) — nothing could build/test before this.
- [x] Make config loading hermetic in tests via an injectable `configHome`
      (2026-06-22).
- [ ] **Pay down typecheck debt** (surfaced by the `typecheck` script). Full
      `tsc --noEmit` count was **397** on 2026-06-22; now **244**. All source
      files except `app.ts` are clean; `app.ts` down to **15** (was 63) via the
      typed `handle()` result map. Fix one module/family per run, no mass-rewrite:
  - [x] `src/capture/` (trajectory-store.ts, replay-service.ts) — DONE run 2.
  - [x] `src/index.ts` (6) — DONE run 3 (barrel alias for cross-module dupes).
  - [x] `src/cli/config.ts` (6) — DONE run 3 (`resolveMergedConfig` helper).
  - [x] `src/control-plane/server.ts` (4) — DONE run 4.
  - [x] `src/orchestrator/operator-runtime.ts` (4 + cascade) — DONE run 4.
  - [~] `src/cli/app.ts` (63 → 15) — **last source file**. Root cause: untyped
    `server.handle()` results. Runs 5–6 added `ControlPlaneResultMap` + typed
    `handle<M>` overload and seeded **cron**, **pairing**, **monitors.***,
    **tasks.stop**, and the direct-builder **sessions** methods
    (remoteStatus/remoteInventory/platformInventory/platformStatus/remoteRepair).
    Remaining 15 errors, now a mix (no longer one root cause):
    - [ ] `sessions.platformControl` + `sessions.remoteControl` (10 errors) —
      return *composed* objects `ok({ ...status, action, results, … })`. Define a
      named exported result type for each (e.g. `SessionRemoteControlResult`) and
      map to it; can't use a bare `ReturnType<>` because the shape is inline.
    - [ ] `BrowserPushDeliveryTarget | { kind:"webhook"; url } ` union (4 errors,
      ~L1041-1099): narrow on `.kind === "webhook"` before reading `.url`.
    - [ ] `WritableStream` vs `Writable` stream-type mismatch (~L1172).
  - [ ] Map-coverage test: assert every `case "x.y":` in `handle`'s switch has a
    `ControlPlaneResultMap` entry or is explicitly allow-listed as `unknown`, so
    new untyped RPC methods are caught instead of silently `unknown`.
- [ ] Typed client facade `createControlPlaneClient(server)`: one method per
    mapped RPC with inferred params/results, so call sites read
    `client.cronList()` and unmapped methods are a compile error, not `unknown`.
  - [ ] Test files (bulk, ~284): `server.test.ts` (234), `app.test.ts` (41),
    `session-stream.test.ts` (15), `gateway-transport.test.ts` (15), others.
- [ ] Add a `verify` npm script (`typecheck && build && test`) and have the
      engine run it as a pre-push self-check each cycle.
- [ ] Interim **source-only typecheck gate**: once `app.ts` is green, add
      `typecheck:src` (`tsc --noEmit` filtered to `src/**` minus `*.test.ts`) and
      gate on it now, locking in the source-clean state without waiting for the
      ~284 test-file errors to clear.
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
- [ ] Pluggable local-model backend interface for the training runner with a
      deterministic mock backend (so cloud/CI tests pass) and a documented seam
      for a real on-device small model.
- [ ] Synthetic event-stream generator to validate capture→dataset→replay
      round-trips without real OS input.
- [ ] Generalization eval harness: measure replay fidelity on held-out but
      related synthetic trajectories.

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
