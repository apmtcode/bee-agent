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
device/os/browser adapters, consent store, ingestion), `src/training/`
(exporter, job store/manifest, runner, execution service), and — new as of
run 9 — `src/movement/` (the model layer: low-level event schema, tokenizer,
synthetic generator, pluggable learnable backend, eval harness).
- [x] **Inventory** capture/training vs. the objective's five pieces (run 9).
      Finding: `capture`/`training` covered *high-level* trajectory spans
      (summarized observations/actions) + shelled training out to mlx/axolotl,
      but the **model layer was entirely missing** — no low-level movement
      schema, no learnable repeat/generalize model, no synthetic streams, no
      eval. `src/movement/` fills exactly that gap.
- [x] **Pluggable local-model backend interface** + deterministic reference
      backend (run 9). `MovementModelBackend` seam in `src/movement/backend.ts`;
      `MarkovMovementBackend` (variable-order Markov + stupid-backoff) is a
      dependency-free, fully deterministic learner for cloud/CI. Real on-device
      small-model backends implement the same interface.
- [x] **Synthetic event-stream generator** (run 9) — `src/movement/synthetic.ts`,
      seeded PRNG (mulberry32), parametrized tasks (open-and-type / menu-select /
      scroll-and-click). Validates capture→dataset→train→replay round-trips with
      no OS input.
- [x] **Generalization eval harness** (run 9) — `src/movement/eval.ts`:
      teacher-forced next-step accuracy + free-running replay fidelity (LCS
      overlap) on held-out trajectories. Tests prove perfect self-replay
      (objective 2c) and recomposition of learned sub-movements on a
      novel-but-related held-out trajectory (objective 2d).
- [ ] **Coordinate-abstracted tokenization** so the model generalizes to *new*
      screen locations, not just new recombinations of seen cells. Predict
      pointer targets *relative to task anchors* (e.g. "to focused element")
      instead of absolute grid cells, so a held-out target in an unseen cell
      still replays. Current backoff generalizes structure but reuses familiar
      coordinates.
- [ ] **Persist/load a trained movement model** (snapshot() → JSON on disk) and
      wire it into `src/training/runner.ts` as a first, real, on-device-free
      training target alongside the mlx/axolotl launch plans.
- [ ] **Bridge high-level `TrajectorySpan` ↔ low-level `MovementEvent`**: derive
      movement datasets from recorded capture spans (device-adapter gestures →
      movement events) so the two halves of the subsystem share one pipeline.

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
- [ ] **Fix the flaky background-task recovery test** (surfaced run 9). Under
      parallel `vitest run`, `operator-runtime.test.ts` (and sometimes
      `background-tasks.test.ts`) intermittently throw a JSON parse error in
      `readState`/`readJsonFile` ("Expected ',' or '}' … position 311"). Fails on
      clean HEAD too — pre-existing, ~1–3 failures depending on scheduling.
      `writeState` uses atomic temp+rename and each task has a `randomUUID`
      baseDir, so the corruption source is non-obvious; needs dedicated
      reproduction under load (suspect: `writeJsonAtomic`'s module-global
      `writeCounter` temp-name scheme, or a shared path under concurrency).
      Making the suite deterministically green unblocks the `verify` pre-push gate.
