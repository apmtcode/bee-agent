# bee-agent self-evolution log

This file is the running memory of the autonomous self-evolution engine. Each
run appends a dated entry: what was audited, what changed, test results, and at
least one new idea. Newest entries first.

---

## 2026-07-16 (run 9) — 🐛 Fix background-task state corruption (`shellQuote`) + de-flake real-subprocess tests

**Audited:** Repo health. Baseline `npm test` was **red** (3–4 failing, and the
count *varied between runs* — a raciness tell) even though run 8 logged 174/174.
`typecheck:src` and `build` were green. Root-caused the failures instead of
papering over them.

**Genuine correctness bug found & fixed** (`src/harness/background-tasks.ts`):
`shellQuote()` escaped an embedded single quote as `"'"'"'` (6 chars, a stray
leading `"`) instead of the POSIX-correct `'"'"'` (5 chars). **Any background
task whose command or cwd contained a single quote produced invalid state JSON**
— which then made the completion writer's `json.loads()` throw under
`set -euo pipefail`, so the task's state stayed stuck at a corrupt "running"
document and every concurrent reader (`sync`/`recover`/monitor polling,
`getBackgroundTaskExecutionState`) crashed with `SyntaxError: Expected ',' or
'}'`. Verified end-to-end with a live spawn repro (torn/invalid reads → 0).
Fix: correct the escape sequence. This is the actual defect that made the
`printf 'line-1\nline-2\n'` fixtures fail.

**Additional reliability hardening (additive):**
- **Atomic state writes in the launch script.** The initial write was
  `printf … | sed … > state.json` (truncate-and-stream in place) and the Python
  completion writer used `write_text` — both non-atomic, so a concurrent reader
  could observe a torn document. Now both write to a unique temp file and
  `mv`/`os.replace` into place (atomic rename on the same filesystem).
- **Fixed the `sed` pid substitution.** `s/"$$"/…/` broke out of the
  double-quoted sed argument, so the running-state `pid` was left as the literal
  string `"$$"` (never the numeric PID) — which then failed every
  `isProcessRunning(pid)` liveness check. Replaced with a bare-token placeholder
  `__OPENCLAW_PID__` (same mechanism the timestamp already uses).
- **`readJsonFile` defense-in-depth** (`src/shared/fs.ts`): bounded retry (4×,
  short backoff) on `SyntaxError` only, so any residual mid-write read from a
  non-atomic external writer resolves instead of crashing; genuinely corrupt
  files still surface the error.

**De-flaked the integration tests (root cause, not masking).** `server.test.ts`,
`operator-runtime.test.ts`, and (via cross-file pollution) `app.test.ts` each
spawned **real** `printf`/`sleep`/`tail -f` subprocesses while *also* driving
execution state explicitly via `writeState`/`writeOutput` and mocking
`isProcessRunning`. The real subprocesses' async state writes raced (and
clobbered) the explicit writes, so assertions flaked on subprocess scheduling
(e.g. `active`↔`degraded`, breaker `failureCount` 2↔3, recover `unchanged`↔
`missing-process`). Any test that mocks liveness clearly intends deterministic
control, so it should also mock the spawn: added a no-op
`backgroundTaskSpawnProcess: () => ({ pid, unref(){} })` to those four runtimes.

**New regression test** (`background-tasks.test.ts`): starts a real task whose
command contains single quotes and asserts the state file always parses, ends
`completed` with `exitCode 0` and a **numeric** pid, and the command
round-trips byte-for-byte.

**Test results:** `typecheck:src` ✅ (exit 0), `build` ✅. `npm test` ✅
**175/175, verified green across 7 consecutive full-suite runs** (was flaky
3–4-failing). Net: +1 test, previously-failing operator-runtime/server tests now
deterministic, and the intermittent app.test pollution eliminated.

**New idea:** the real-subprocess-in-unit-tests hazard is systemic — several
other runtimes (`gateway-transport.test.ts`, `session-stream.test.ts`) mock
`isProcessRunning` and could latently spawn real processes. Add a tiny test
helper `deterministicBackgroundTaskOptions()` (returns both the no-op spawn and
`isProcessRunning`) and a lightweight lint/grep guard in CI that flags any test
runtime mocking one without the other, so this class of flakiness can't
reappear. Longer term: give `StandaloneOperatorRuntime` a documented
"in-process executor" backend (no shell/OS spawn at all) so background-task
logic can be unit-tested hermetically, reserving the real shell launcher for a
small number of explicit integration tests.

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
