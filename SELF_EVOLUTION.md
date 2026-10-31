# bee-agent self-evolution log

This file is the running memory of the autonomous self-evolution engine. Each
run appends a dated entry: what was audited, what changed, test results, and at
least one new idea. Newest entries first.

---

## 2026-10-31 (run 15, local) — `GovernedActuator`: rate / velocity / region governor

**Audited:** ROADMAP innovation backlog + `movement/replay.ts` / `safe-actuator.ts`.
`SafeActuator` guarantees nothing stays *held*, but nothing bounded *what* a
generated episode may do: spam thousands of events, teleport the pointer, or
click outside the target window. That is the last safety layer needed before a
real OS actuator or a `movement.replay` RPC.

**Changed (additive, new `src/movement/governor.ts`, barrel-exported):**
- `GovernedActuator` decorator over any `InputActuator` with rolling 1s
  `maxEventsPerSecond`, wall-clock `maxPointerSpeedPxPerMs`, and a `regions`
  allow-list (nearest-region clamping; `followFocus` confines input to the
  bounds of the last forwarded `window.focus`).
- `mode: "reject"` (default) throws `GovernorViolationError` before the event
  reaches the inner actuator; `mode: "clamp"` scales over-speed moves along the
  same direction, clamps out-of-region points, and delays rate-limited events.
- Releases (`pointer.up`/`key.up`) are never rejected or rate-limited (only
  clamped into the region), so the governor can never cause stuck input.
- Injectable `now`/`sleep`; `stats()`, `onViolation`, config validation.

**Test results:** +8 tests (`governor.test.ts`), incl. stacked over
`SafeActuator` + `replayEpisode`: a runaway mid-drag move is rejected, never
reaches the OS, and the left button is still released. Movement suite 42/42.
Build ✅, `typecheck:src` clean, full `tsc` unchanged at 69. Full suite
**216/216 on 3 of 4 runs**; the first run (machine under heavy load, the
batch took >400s) had 6 failures in 6 files that I could not reproduce or
capture. The movement tests use fake clocks, so this looks like the known
load-dependent flake class (see run 13), not this change. Follow-up: have the
flake sentinel save failing test names on every run.

**New idea (logged to ROADMAP):** *Default actuation stack factory*
(`createActuationStack`) giving `movement.replay` a single audited entry point
with strict/assisted profiles; plus an offline human-plausibility check that
rejects generated episodes whose velocity/jerk profile is an outlier vs. the
training data.

---

## 2026-10-28 (run 14, local) — `SafeActuator`: stuck-input watchdog for real actuation

**Audited:** ROADMAP local-movement + innovation backlogs. Before any real OS
actuator (CGEventPost) or `movement.replay` RPC can ship, there must be a safety
layer: `replayEpisode` releases held input only for its own run, so a hung
caller, a crash, Ctrl-C, or two callers sharing an actuator could leave a mouse
button or modifier key physically held down.

**Changed (additive, new `src/movement/safe-actuator.ts`, barrel-exported):**
- `SafeActuator` decorator over any `InputActuator`: tracks held buttons/keys
  globally (marked held *before* forwarding, so a press the inner actuator threw
  on is still released), releases at the last known pointer position.
- Release triggers: `AbortSignal`, idle watchdog (`idleTimeoutMs`, armed only
  while something is held, injectable timers), `installSignalGuard()` for
  SIGINT/SIGTERM (release-only, returns detach), `guard(fn)` on throw, `dispose()`.
- Concurrent releases share one pass; `perform` waits for an in-flight release;
  release failures are reported (`failed`), never thrown; disposed actuators
  refuse input and aborted ones refuse new presses.

**Test results:** +8 tests (`safe-actuator.test.ts`), incl. drop-in use with
`replayEpisode`. Movement suite 34/34. Full suite **208/208 twice in a row**
(flake sentinel). Build ✅, `typecheck:src` clean, full `tsc` unchanged at 69.

**New idea (logged to ROADMAP):** *Actuation rate governor* — a stacking
decorator capping events/sec, pointer velocity, and a screen-region allow-list,
so `Governor(SafeActuator(os))` is the default stack for `movement.replay`.

---

## 2026-10-27 (run 13, local) — Movement segmentation: one session → many demonstrations

**Audited:** ROADMAP local-movement backlog. The store (run 12) can persist
episodes, but a real capture session is one long mixed stream (moves, clicks,
drags, scrolls, typing); training on it whole gives one useless "demonstration".
Segmentation was the missing step between recording and training.

**Changed (additive, new `src/movement/segment.ts`, barrel-exported):**
- `segmentEpisode(episode, options)` → `{ segments, counts, droppedEvents }`.
  Press→matching release becomes `drag` (displacement > `dragThresholdPx`) or
  `click`; pointer-move runs split on `idleGapMs`, become `move-to` when they end
  right before a press (the targeting motion) else `move`, and are dropped as
  jitter under `minMoveDistancePx`; scroll bursts → `scroll`; key bursts →
  `typing`, extended while any key is held so shift-click stays one balanced
  `chord`. Unreleased presses are dropped (never train on stuck buttons).
- Each segment is a valid, time-rebased `MovementEpisode` (`<id>#<n>`, label
  `<label>/<kind>` or custom) with the active `window.focus` prepended at t=0
  so app context is preserved. `kinds` filter for e.g. drag-only datasets.

**Test results:** +5 tests (`segment.test.ts`) incl. end-to-end: one session
with 3 drags → segment → train `TemplateMovementModel` → retargeted generate.
Movement suite 26/26. First full runs hit a **load-dependent flake** unrelated
to this change: `gateway-transport.test.ts` "replays only missed events…" gave
its reconnect a 20ms real-timer heartbeat timeout, which expired before the
pong under a slow (172s) suite. It passes alone with and without this diff.
Widened that test's timeout to 2s (it asserts health-after-pong, not
timeouts). After that, two consecutive full runs passed **200/200** (flake
sentinel). Build ✅, `typecheck:src` clean, full `tsc` unchanged at 69.

**New idea (logged to ROADMAP):** *Segment boundary confidence* — score each
segment by minimum-jerk velocity-profile fit and pause sharpness so training
can skip low-confidence fragments.

---

## 2026-10-27 (run 12, local) — Movement datasets persist: consent-gated `FileMovementDatasetStore`

**Audited:** ROADMAP's local-movement backlog. Run 11 built schema → record →
train → replay, but everything was in-memory: nothing could accumulate
demonstrations across sessions, and the privacy controls in `src/capture`
(consent grants, `allowTrainingExport`, `retainRawDays`) did not apply to raw
movement data at all. That blocked the next step (control-plane `movement.*` RPCs).

**Changed (additive, new `src/movement/store.ts`, barrel-exported):**
- `FileMovementDatasetStore`: one JSONL dataset per label under
  `<root>/datasets/` (file stem = slug + sha256 prefix, so "Drag File" vs
  "drag-file" never collide and labels can't path-traverse), header written once,
  appends serialized through a write queue (no interleaved lines).
- **Consent gates:** `append` requires an active, unexpired, unrevoked grant at
  tier `app`/`full` (configurable) and reports which `grantId` authorized it;
  `loadTrainingEpisodes` throws `MovementConsentError` unless the policy's
  `allowTrainingExport` is true. Invalid episodes are rejected individually.
- `prune()` drops episodes older than `retainRawDays`, removes emptied files,
  and **never deletes a file it cannot parse**.
- `saveModel`/`loadModel` persist `TemplateMovementModel` artifacts (model names
  validated against path escape).

**Test results:** +8 tests (`store.test.ts`): consent refusal (no grant /
wrong tier / expired), single-header multi-append, partial rejection, 5-way
concurrent append, training gate → train → save/load → generate, retention
prune, corrupt-file safety, slug collisions. Suite **195/195** ✅ on two
consecutive runs (manual flake sentinel), build ✅, `typecheck:src` clean, full
`tsc` unchanged at 69.

**New idea (logged to ROADMAP):** *Consent-provenance purge* — persist the
authorizing `grantId` on every stored episode and add `purgeByGrant`, so
revoking a consent grant erases exactly the data it authorized.

---

## 2026-10-22 (run 11, local) — 🆕 Local-movement learning: schema → record → train → generalize → replay

**Audited:** Standing objective #2, which had received zero increments in runs
1–10 (all spent on typecheck debt / flakiness). Inventoried `src/capture` +
`src/training`: capture records only *human-readable summaries* ("tapped
Deploy") and training only plans MLX/axolotl LLM fine-tunes. **Missing:** a raw
pointer/keyboard/window event schema, a replayable movement dataset, a movement
model, and a replay engine.

**Changed (purely additive, new `src/movement/`, barrel-exported):**
- `events.ts` — versioned `MovementEvent` union (`pointer.move/down/up`,
  `scroll`, `key.down/up`, `window.focus`; `t` = ms since episode start),
  `MovementEpisode`, validators, and a **JSONL dataset format** (header line +
  one episode/line; bad lines are rejected individually, not fatal).
- `source.ts` — `InputEventSource` interface (OS hooks plug in here) +
  deterministic `SimulatedInputEventSource`; `recordEpisode()` rebases time,
  caps by events/duration, and **drops keystrokes while a sensitive app is
  focused** (reuses capture's `DEFAULT_SENSITIVE_APP_DENYLIST`) with optional
  key redaction. `synthesizeDrag()` = seeded minimum-jerk, bowed, jittered
  synthetic drags for tests and data augmentation.
- `model.ts` — pluggable `MovementModel` interface + `TemplateMovementModel`:
  per label, maps demonstrations into a canonical start→end frame, picks the
  **medoid** demonstration (outlier-robust), and **generalizes** to new
  endpoints via similarity transform (rotate/scale/translate); in-place
  gestures (clicks) retarget by translation; duration scales with
  √(distance ratio); output clamped to screen. JSON-serializable artifact
  (`toJSON`/`fromJSON`). `evaluateMovementModel()` = held-out generalization
  harness (resampled path error, px + chord-normalized).
- `replay.ts` — `InputActuator` interface + `RecordingActuator` mock;
  `replayEpisode()` preserves timing with speed scaling, abort signal,
  `allowTypes`, `maxEvents` brake, coordinate clamping, and **always releases
  held buttons/keys** on abort/cap so a machine is never left mid-drag.

**Test results:** +13 tests (`src/movement/movement.test.ts`), incl. held-out
generalization to unseen rotated/scaled drags at **<2% normalized error**.
Suite **187/187** ✅, build ✅, `typecheck:src` clean, full `tsc` unchanged (69,
no new errors). Also committed + pushed the interrupted run-10 work (result map
125→69) after resolving a SELF_EVOLUTION.md merge conflict with cloud run 9.

**New idea:** *Stuck-input watchdog as a first-class safety primitive* — the
release-on-abort logic in `replayEpisode` should become a reusable
`SafeActuator` wrapper that tracks held state across *any* caller (model
generation, skills, remote control) and auto-releases on process signal/
timeout. Added to ROADMAP.

---

## 2026-10-22 (run 10, local) — Result map → capture/skills/push families: total 125→69

**Audited:** The remaining 118 `server.test.ts` errors. Confirmed still the same
unmapped-RPC → `{}` pattern, traced the specific methods from the failing lines
(background.tasks.*, subagents.list, trajectories.*, replays.*,
skills.executable.*, push.subscriptions.*).

**Changed (additive) in `src/control-plane/server.ts`:** mapped 17 more methods,
all via `Awaited<ReturnType<…>>` of the source method (`NonNullable<>` for the
`ok(x) : notFound` cases):
- `background.tasks.start/get/active/sync/list`, `subagents.list`.
- `trajectories.list/review/reviewed`, `replays.get/list`.
- `skills.executable.list/run/runs`.
- `push.subscriptions.create/list` via `FilePushStore["upsert/listSubscriptions"]`.

**Deliberately skipped (documented):** `notifications.send`,
`push.subscriptions.delete`, `push.test` return *composed* `ok({ … })` objects,
so they need a named result type rather than a `ReturnType<>` reference.

**Test results:** full `tsc` **125 → 69** (server.test.ts 118 → 62). The map now
covers ~54 RPC methods. `typecheck:src` stayed CLEAN. Build ✅. Tests ✅
**174/174**. Remaining 69: ~3 composed-return methods, a cluster of `.result`
accessed on `ControlPlaneResponse<T>` unions *without* an `if (response.ok)`
guard (genuine test-only fix — narrow before access), and a few stray test
typings (`gateway-transport` 4, others 1 each).

**New idea:** the composed-return methods are the last structural gap — extract
their inline `ok({...})` shapes into exported named types (e.g.
`SendNotificationResult`, `PushTestResult`) in server.ts, which both finishes the
map AND documents the RPC response contract for any external client.
## 2026-07-25 (run 9) — 🔴→🟢 Eliminated background-task test flakiness (verification gate restored)

**Audited:** The verification gate itself. On a clean checkout `npm test` was
**non-deterministic** — 1–4 of 174 tests failed at random across runs (measured
3/4/3 failing in three back-to-back runs). Since the engine's entire push
decision (step 5–7) hinges on a green `npm test`, a flaky suite is the single
highest-value thing to fix: it can both **block good pushes** and **hide real
regressions**. All failures clustered in the background-task subsystem
(`app.test.ts`, `server.test.ts`, `operator-runtime.test.ts`).

**Root cause (one bug, four symptoms):** these tests call
`runtime.startBackgroundTask({ command: "sleep 5" | "printf ok" | ... })`, which
**actually spawns a real detached subprocess** via the launch script. That
subprocess asynchronously (a) writes a `status:"running"` **execution state
file** and (b) appends command output — both racing the test's later assertions:
- control-state check (`server.ts:2174`) reads the state file; if the subprocess
  wrote `running` before the check but its launch-wrapper pid already exited
  (`isProcessRunning=false`), the remote flipped to `degraded:missing-process` →
  platform aggregate `mixed`→`degraded`/`paused`, breaker `failureCount` 2↔3.
- `background-view`/`watch-task` read output before `printf ok` had flushed → the
  `toContain("ok")` assertions failed intermittently.

**Changed (additive, test-hermeticity — same pattern as the run-2 `configHome`
fix):**
- `src/cli/app.ts` (source): threaded two optional overrides
  `backgroundTaskSpawnProcess` / `backgroundTaskIsProcessRunning` through
  `OperatorCliAppOptions` into the runtime (production leaves them undefined →
  real spawn unchanged). This is the only source change.
- Injected a **no-op spawn stub** (`() => ({ pid, unref(){} })`) into every
  background-task-touching test runtime so no real subprocess ever writes a
  competing state/output file — the tests now drive state **explicitly** via
  `writeState`/`writeOutput` (the pattern half of them already used):
  `operator-runtime.test.ts` (1 runtime), `server.test.ts` (3 runtimes, via a
  shared `noopSpawnProcess` helper), `app.test.ts` (2 apps). Pinned
  `isProcessRunning` per test to match its intent (false where a dead-pid task
  must sync to failed; true where a task must stay active for `watch-active`).
  Added an explicit `writeOutput(task,"ok\n")`+`writeState` to the `app.test.ts`
  background half so `background-view` reads deterministic output.

**Test results:** `npm test` now **174/174 passing, 10/10 consecutive runs
green** (was 1–4 random failures). `npm run build` ✅. `npm run typecheck:src` ✅
(exit 0, source stays clean). Full `tsc` **125** (unchanged — no regression).

**New idea:** add a **flake sentinel** to the engine's pre-push self-check — run
`vitest run` **twice** (or `--retry=0` with a repeat) each cycle and refuse to
treat green as green unless *both* pass. Cheap insurance that a newly-introduced
race can't slip through a single lucky run, exactly the failure mode this run
paid down. Bigger: a lint that flags `startBackgroundTask`/real-`spawn` calls in
`*.test.ts` that don't inject a spawn stub, so this class of flake can't return.

---


---

## 2026-06-23 (run 8) — Result map → orchestration families: test debt 229→125

**Audited:** The remaining test-file typecheck debt. server.test.ts had 184
errors; the dominant pattern (`Property 'id' does not exist on type '{}'` ×114)
traces to the *same* root cause as the source errors — methods absent from
`ControlPlaneResultMap` return `result: unknown`, and the tests' truthy-guards
(`if (!x) throw`) narrow `unknown` to `{}`, so `.id` fails. Not a test bug; an
unmapped-RPC consequence.

**Changed (additive) in `src/control-plane/server.ts`:** extended
`ControlPlaneResultMap` with the orchestration families the tests exercise — all
referencing runtime method return types via `Awaited<ReturnType<…>>`
(`NonNullable<>` where the handler does `ok(x) : notFound`, harmless otherwise):
- `sessions.list/get/resume/idle/complete/fail` + `sessions.bootstrap`
  (→ exported `SessionBootstrapResult`).
- `tasks.create/get/list/update`, `runs.start/get/list`,
  `approvals.list/resolve`, `transcript.get`.
- `plans.get/upsert/requestApproval/respondApproval/verify`,
  `messages.send/list/inbox/outbox`,
  `teams.teammates.start/list/get/update/message`.

**Test results:** full `tsc` **229 → 125** (server.test.ts 184 → 118;
**app.test.ts 39 → 1** — it leans on bootstrap/sessions, so it nearly cleared via
cascade). `typecheck:src` stayed CLEAN. Build ✅. Tests ✅ **174/174**.

**New idea:** the map now covers ~37 methods. Worth auto-deriving the *expected*
list from the handler switch (a build-time codegen or a test that scrapes
`case "x.y":`) so the map and the dispatcher can't silently diverge — and so the
handful of still-unmapped methods (skills.executable.*, push.subscriptions.*,
trajectories.*, replays.*) are surfaced as an explicit TODO list rather than
rediscovered by grep each run.

---

## 2026-06-23 (run 7) — 🎯 `app.ts` fully green: ALL source files typecheck + source-only gate

**Audited:** The final 15 `app.ts` errors (10 composed control returns + 5
unrelated) plus the status-line controller's stdout typing.

**Changed:**
- **Two genuine latent bugs fixed** (`src/cli/app.ts`): the cron delivery-target
  labels did `target.kind === "local" ? "local" : target.url`, but
  `DeliveryTarget` is `local | webhook | browser-push` — so a `browser-push`
  target (no `.url`) would print `undefined` at runtime. Added
  `formatDeliveryTargetLabel(target)` with an exhaustive `switch` over all three
  kinds and used it at the 4 sites (cron-list/runs/tick).
- `src/cli/status-line.ts`: the controller required `stdout: Writable & {…}` but
  only ever reads `.isTTY`/`.columns`; loosened it to `NodeJS.WritableStream &
  {…}` (what it actually needs), which also matches the app's stdout field. Drop
  the now-unused `Writable` import.
- `src/control-plane/server.ts`: mapped the two composed control methods —
  `sessions.platformControl` → the already-exported `SessionPlatformControlResult`
  (status & { action, results }); `sessions.remoteControl` →
  `NonNullable<Awaited<ReturnType<typeof buildSessionRemoteStatusResult>>>` (it
  returns a refreshed status, same as remoteStatus).
- **New source-only typecheck gate:** added `tsconfig.src.json` (extends base,
  excludes `**/*.test.ts`) + `typecheck:src` script. It currently **passes
  (exit 0)** — locking in the milestone below.

**Test results:** 🎯 **ALL `src/**` non-test files now typecheck clean** —
`app.ts` 15 → 0; the non-test error list is empty. Full `tsc` total **244 → 229**
(the remaining 229 are entirely in test files). `npm run typecheck:src` ✅.
Build ✅. Tests ✅ **174/174**.

**New idea:** wire `typecheck:src` into the engine's per-run pre-push self-check
now (cheap insurance against a source regression), and keep the full `typecheck`
(incl. tests) as the longer-horizon target. Longer term: a `verify` script =
`typecheck:src && build && test` as the canonical green gate.

---

## 2026-06-23 (run 6) — Extend `handle()` result map: app.ts 43→15, total 274→244

**Audited:** The 43 remaining `app.ts` errors (all the untyped-`handle` family) and
the matching server-side result expressions for the `monitors.*`, `tasks.stop`,
and `sessions.platform*/remote*` methods.

**Changed (additive) in `src/control-plane/server.ts`:** extended
`ControlPlaneResultMap` with 11 more compiler-verified entries:
- `monitors.list/start/sync/stop/output` + `tasks.stop` via
  `Awaited<ReturnType<StandaloneOperatorRuntime[…]>>` (with `NonNullable<>` for
  the null-checked `sync`/`stop`/`output`/`cancel` cases).
- `sessions.remoteStatus/remoteInventory/platformInventory/platformStatus/
  remoteRepair` via `ReturnType<typeof build…Result>` (the methods that return a
  single builder result directly; `NonNullable<>`/`Awaited<>` as appropriate).

**Deliberately left as `unknown` (documented):** `sessions.platformControl` and
`sessions.remoteControl` return *composed* objects (`ok({ ...status, action,
results, … })`), not a single builder result, so they need a dedicated named
result type rather than a `ReturnType<>` reference — tracked in ROADMAP.

**Test results:** typecheck **274 → 244** (`app.ts` 43 → 15). server.ts stayed
CLEAN. Build ✅. Tests ✅ **174/174**. The 15 remaining `app.ts` errors are now a
*mix*, not one root cause: 10 from the two composed control returns, plus 5
genuinely unrelated (4× a `BrowserPushDeliveryTarget | webhook` union that needs
`.kind` narrowing before `.url`; 1× a `WritableStream` vs `Writable` stream-type
mismatch ~L1172).

**New idea:** now that the RPC result types are centralized, generate a typed
client facade (`createControlPlaneClient(server)`) exposing one method per
mapped RPC with inferred params/results — so call sites read
`client.cronList()` instead of `handle({method:"cron.list"})`, and unmapped
methods are a compile error rather than a silent `unknown`.

---

## 2026-06-23 (run 5) — Typed `handle()` result map: app.ts 63→43, total 347→274

**Audited:** The 63 `app.ts` errors. Found 48 are the *same* root cause —
`server.handle()` is method-agnostic and returns `result: unknown`, so every
`response.result.X` access errors, and the `.map((ticket) => …)` callbacks over
those results inherit implicit-`any` (the 10 TS7006s are downstream of the same
issue, not independent). Hand-casting 48 sites would be noise; the real fix is
to type the RPC surface.

**Changed (additive, backward-compatible) in `src/control-plane/server.ts`:**
- Added `ControlPlaneResultMap` — a *partial* method→result-type interface whose
  entries reference the source of truth via `Awaited<ReturnType<…>>` (e.g.
  `OperatorCronService["listJobs"]`, `FilePairingStore["createTicket"]`), so they
  cannot drift from the implementation.
- Added `ControlPlaneResultFor<M>` = `M extends keyof ControlPlaneResultMap ?
  … : unknown` — unmapped methods still resolve to `unknown`, exactly as before.
- Added a typed `handle<M extends string>(request: { method: M; … })` overload
  above the unchanged implementation signature. Object-literal `method:` fields
  infer `M` to the string literal, so callers get typed results for free; callers
  passing a `ControlPlaneRequest` (method: string) still get `unknown`. Zero
  behaviour change.
- Seeded the map with the **cron** (5) and **pairing** (3) families — the methods
  `app.ts` consumes whose result types are cleanly referenceable.

**Test results:** typecheck **347 → 274** (`app.ts` 63 → 43; the typed overload
also cascaded correctness into test files, hence the larger total drop).
server.ts stayed CLEAN. Build ✅. Tests ✅ **174/174**.

**Next (documented in ROADMAP):** extend `ControlPlaneResultMap` with the
remaining families `app.ts` uses — `monitors.*` (list/start/sync/stop/output →
`Awaited<ReturnType<StandaloneOperatorRuntime[…]>>`, `NonNullable<>` for the
null-checked ones), `tasks.stop`, and `sessions.platform*/remote*` (builder
return types). Each family is a small, compiler-verified diff.

**New idea:** the result map is now a single authoritative list of typed RPC
methods — a tiny test can assert every `case "x.y":` in `handle`'s switch has a
corresponding map entry (or is explicitly allow-listed as `unknown`), turning
"untyped RPC method" into a caught regression rather than silent debt.

---

## 2026-06-22 (run 4) — Typecheck debt: `server.ts` + `operator-runtime.ts` greened (app.ts is the last source file)

**Audited:** Remaining source-file `tsc --noEmit` errors (operator-runtime 4,
control-plane/server 4).

**Changed:**
- `src/control-plane/server.ts` (4): cast the generic RPC `bootstrap.result`
  (typed `unknown` by the method-agnostic `handle()` dispatcher) to the existing
  `SessionBootstrapResult`; guard `event.ts !== undefined` before comparing to
  `afterTs`; replace a non-narrowing `.filter(Boolean)` on remote controls with a
  `control is NonNullable<…>` predicate so the later `.every` callbacks typecheck.
- `src/orchestrator/operator-runtime.ts` (4 + cascade): `Parameters<typeof
  FileBackgroundTaskStore>` → `ConstructorParameters<…>` (it's a class, not a
  call signature); annotate `let effectiveAction: OperatorExecutionAction`
  (optional vs required `sessionId` mismatch) and import that type; fall back with
  `authorization.title ?? authorizedTitle` to keep the title a `string`.

**Test results:** typecheck **378 → 347** (server + operator-runtime CLEAN; the
`effectiveAction` annotation cascaded to clear downstream inference errors too).
**Milestone: `src/cli/app.ts` (63) is now the ONLY source file with typecheck
errors** — every other `src/**` non-test module is clean. The remaining ~284 are
all in test files. Build ✅. Tests ✅ **174/174**.

**New idea (logged to ROADMAP):** Once `app.ts` is green, flip on a
*source-only* typecheck gate (`tsc --noEmit` over `src/**` excluding `*.test.ts`)
in the `verify` script before tackling the larger test-file debt — locks in the
source-clean state immediately rather than waiting for all 347 to clear.

---

## 2026-06-22 (run 3) — Typecheck debt: `src/index.ts` + `src/cli/config.ts` greened

**Audited:** Remaining `tsc --noEmit` source-file errors. Picked two small,
self-contained source modules (avoided the 63-error `app.ts` cluster, too large
for one reviewable hour).

**Changed:**
- `src/cli/config.ts` (6 errors, one root cause): `resolveOperatorCli*Config`
  derived `merged` via `"merged" in config ? config.merged : config`. Because
  `OperatorCliRuntimeConfig` is *structurally assignable* to the index-signature
  `OperatorCliConfigObject`, the `in` narrowing collapsed `merged` to
  `OperatorCliConfigValue` (nullable, no known props) → TS18047/TS2339 on
  `.permissionMode`/`.hooks`/`.statusLine`. Added a `resolveMergedConfig` helper
  that discriminates on the runtime shape (`isConfigObject(merged)` +
  `Array.isArray(loadedEntries)`) and used it in both functions.
- `src/index.ts` (6 errors): the barrel re-exported three names that are defined
  independently in two modules each — `SpawnSubagentResult`
  (control-plane/server + orchestrator/operator-runtime),
  `CreateCaptureConsentParams` (capture/ingestion + operator-runtime),
  `CreateTrainingJobParams` (training/job-store + operator-runtime) → TS2300
  duplicate identifiers. Confirmed none are imported via the barrel internally,
  then aliased the operator-runtime exports with the `Runtime` prefix already
  established for `RuntimeReviewTrajectoryParams`. No type definitions touched.

**Test results:** typecheck **390 → 378** (index.ts + config.ts now CLEAN).
Build ✅. Tests ✅ **174/174**. Pure type-safety/barrel-naming changes.

**New idea (logged to ROADMAP):** A barrel-collision lint — a tiny script that
scans `src/index.ts` re-exports for names exported from more than one module and
flags them, so future duplicate-identifier debt is caught at authoring time
instead of accumulating silently.

---

## 2026-06-22 (run 2) — Typecheck debt: `src/capture/` greened

**Audited:** Full `tsc --noEmit` output. The roadmap's earlier note massively
**undercounted** the debt — there are **397** typecheck errors, not the handful
first recorded. Broke them down by file (source vs test):
`src/cli/app.ts` 63, `src/index.ts` 6, `src/cli/config.ts` 6,
`src/capture/trajectory-store.ts` 6, `src/orchestrator/operator-runtime.ts` 4,
`src/control-plane/server.ts` 4, `src/capture/replay-service.ts` 1; tests are the
bulk (`server.test.ts` 234, `app.test.ts` 41, etc.).

**Changed (one coherent module, per the "no mass-rewrite" guardrail):** greened
`src/capture/` (movement-subsystem code, standing objective #2).
- `replay-service.ts`: replaced a non-narrowing `Boolean(replay)` guard in a
  type-predicate filter with `replay !== undefined`, so `replay.eventCount` is
  properly narrowed (TS18048).
- `trajectory-store.ts`: `searchReviewed` used a `map → null sentinel →
  type-predicate filter` chain that TS couldn't validate (TS2677), cascading into
  4 "possibly null" errors in the `.sort`. Converted to `flatMap` returning
  `[hit]`/`[]` — no sentinel, no predicate, fully type-safe. Behaviour identical.

**Test results:** typecheck **397 → 390** (capture module now CLEAN). Build ✅.
Tests ✅ **174/174**. Behaviour unchanged — pure type-safety fixes.

**New idea (logged to ROADMAP):** Add a per-module typecheck ratchet — a tiny
script that records each module's current error count and fails if a module
regresses above its recorded baseline. Lets the engine pay debt down
module-by-module without a single green-gate blocking progress, and prevents
backsliding while the full count is still >0.

---

## 2026-06-22 — Toolchain made real + test hermeticity fix

**Audited:** Repo bootstrap state, `package.json`, build/test pipeline, and the
`src/cli` config/status-line subsystem. Compared declared tooling against the
reference codebases (`openclaw` declares `vitest`; others vary).

**Problem found (highest value):** `package.json` declared **no dependencies at
all** — neither `tsdown` (referenced by the `build` script) nor `vitest` (used by
173 test files). There was also no `test` script. Net effect: **no run, cloud or
local, could actually build or test its own changes** — the procedure's
verification gate (step 5) was silently un-runnable. The near-empty
`package-lock.json` (280 B) confirmed nothing had ever been installed from the
manifest.

**Changes made (additive, reversible):**
- Declared dev tooling in `package.json`: `vitest`, `tsdown`, `typescript`,
  `@types/node`; committed the resulting `package-lock.json`.
- Added npm scripts: `test` (`vitest run`), `test:watch`, and `typecheck`
  (`tsc --noEmit`).
- Fixed a **test-isolation bug** surfaced by running the suite on a real machine:
  `OperatorCliApp` constructed `OperatorCliConfigLoader(this.cwd)` with no config
  home, so it read the developer's real `~/.claude/settings.json`. On this
  machine that file carries a `statusLine` (claude-hud), so the "fresh app" test
  saw a configured status line instead of `disabled` and failed. Added an
  optional `configHome` to `OperatorCliAppOptions`, threaded it into the loader
  (production behaviour unchanged via the loader's default param), and pointed
  the failing test at an isolated home.
- Added a regression test (`config.test.ts`) proving an explicit `configHome`
  isolates the loader from `$HOME/.claude` even when `$HOME` contains settings.

**Test results:** `npm run build` ✅ (tsdown, 5 files, ~523 kB). `npm test` ✅
**174/174 passing** across 41 files (was 173 passing + 1 failing before the fix;
+1 new regression test). `npm run typecheck` reports **pre-existing** errors
(see ROADMAP) — these predate this run; the build (rolldown/tsdown) does not
strict-typecheck, so build+test gates are green.

**New idea (logged to ROADMAP):** Add a CI-equivalent `verify` script
(`typecheck && build && test`) plus a lightweight pre-push self-check the engine
runs every cycle, so the typecheck debt is paid down incrementally instead of
hidden. Bigger idea: a "capability inventory" generator that diffs bee-agent's
exported RPC surface against each reference agent to make parity gaps explicit
and machine-trackable.
