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
  - [x] Map-coverage test — DONE run 9 (`result-map-coverage.test.ts`). Scrapes
    every switch `case` + every `ControlPlaneResultMap` key from source; fails on
    any unmapped method or stale allow-list entry. Allow-list is **empty** (100%
    coverage).
- [ ] Typed client facade `createControlPlaneClient(server)`: one method per
    mapped RPC with inferred params/results, so call sites read
    `client.cronList()` and unmapped methods are a compile error, not `unknown`.
    Now low-risk: the map is complete (run 9) and the coverage guard prevents
    drift, so the facade can be generated directly from `ControlPlaneResultMap`.
  - [x] Result map now covers **all 117** dispatched methods (run 9 mapped the
    final ~50: subagents, skills, background.tasks, training, trajectories,
    replays, plugins, capture, push.*, notifications, memory, runs.*).
  - [ ] Residual 15 test-only typings (no longer unmapped-RPC related): 6×
    `server.test.ts` `.result` access without an `ok`-guard (~L1040–1056), 2×
    `run` possibly-undefined (~L1891), `status-line.test.ts` `WritableStream`
    mock shape (L165), `app.test.ts` `run.metadata` (L729), `gateway-transport`
    `unknown`/`event.ts` (4), `session-stream` `runId` shape (1). Small final
    sweep to a fully-green `tsc`.
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
- [ ] Pluggable local-model backend interface for the training runner with a
      deterministic mock backend (so cloud/CI tests pass) and a documented seam
      for a real on-device small model.
- [ ] Synthetic event-stream generator to validate capture→dataset→replay
      round-trips without real OS input.
- [ ] Generalization eval harness: measure replay fidelity on held-out but
      related synthetic trajectories.

## Reliability / correctness (background-task subsystem)
- [x] Fix `shellQuote` POSIX single-quote escaping (`"'"'"'` → `'"'"'`) — DONE
      run 9. Was corrupting the launch-script state JSON for any command
      containing a single quote.
- [x] Make launch-script state writes atomic (temp + rename / `os.replace`) —
      DONE run 9, so concurrent readers never see a torn state file.
- [ ] **Launch-script fuzz test**: render + execute the launch script for a
      matrix of adversarial commands (single/double quotes, `$`, backticks,
      newlines, `%`, unicode) and assert the resulting state file always parses
      and round-trips `command`/`cwd` faithfully. Would have caught the run-9
      `shellQuote` bug directly.
- [ ] Fix the `$$` PID substitution in the initial running-state write: the sed
      `s/"$$"/$$/` treats `$` as a regex EOL anchor and never matches, so the
      initial state records `"pid":"$$"` (a string) instead of the numeric PID.
      Harmless today (reconcile re-derives liveness) but a latent correctness gap
      — move the initial write to the same Python writer used on exit.

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
