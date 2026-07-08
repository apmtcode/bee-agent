# bee-agent self-evolution log

This file is the running memory of the autonomous self-evolution engine. Each
run appends a dated entry: what was audited, what changed, test results, and at
least one new idea. Newest entries first.

---

## 2026-07-08 (run 9) — 🧠 Pluggable movement-model backend (train/repeat/generalize) + real shellQuote bug fix

**Audited:** The local-movement learning subsystem (objective #2). `src/capture/`
captures events and `src/training/` builds *external* launch scripts (mlx/axolotl)
— but nothing actually **trains a local model or runs inference in-process**, so
objectives 2(c) "repeat recorded movements" and 2(d) "generalize" had no runnable,
cloud-testable implementation (the Apple-silicon runtimes can't execute in CI).

**Changed (additive):**
- **New `src/training/movement-model.ts`** — the pluggable local-model seam:
  - `MovementModelBackend` interface (`train(dataset, opts) -> MovementModel`) and
    `MovementBackendRegistry` (register/resolve by id) so a real on-device backend
    swaps in without touching call sites; `createDefaultMovementBackendRegistry()`
    pre-seeds the deterministic mock.
  - `MarkovMovementBackend` (`id: "markov-mock"`) — a dependency-free, deterministic
    back-off n-gram model. **Repeats** recorded sequences exactly via `generate()`
    (2c) and **generalizes** to unseen contexts by backing off to known shorter
    suffixes (2d). No randomness → reproducible in CI.
  - `tokenizeReplayEvent` / `buildMovementDataset` derive token sequences from the
    existing `ReplayManifest` event format; `MovementModel.evaluate()` gives a
    teacher-forced **replay-fidelity** metric (accuracy + backoff rate); full
    JSON `toJSON()`/`restoreMovementModel()` round-trip for persistence.
  - Exported the whole surface from `src/index.ts`.
- **Real reliability bug fixed** in `src/harness/background-tasks.ts` `shellQuote`:
  the POSIX single-quote escape replacement was `"'"'"'` (starts with `"`) instead
  of `'"'"'` (must **close** the open quote first). The stray leading `"` corrupted
  the launch-script state JSON for **any command containing a single quote** (e.g.
  `printf 'x'`): the `pid` sed-substitution silently failed and the `command` value
  was mangled → `readState` threw `SyntaxError` on parse. `runner.ts` already had
  the correct form, so this was a copy divergence. Verified via a spawn repro: state
  now parses, `pid` is a real number, `command` is intact.

**Test results:** new `movement-model.test.ts` **10/10** ✅ (tokenize, dataset,
exact-repeat, backoff-generalize, zero-step fidelity, empty-model, determinism,
serialization round-trip, registry). `typecheck:src` ✅ (exit 0). Build ✅.
The shellQuote fix **deterministically greened `operator-runtime.test.ts`** (was a
JSON-parse crash) and stabilized `app`/`server` (baseline flaked 2→3 failures under
spawn races; now stably 2). Full `npm test`: **181/184**, with **zero regressions
from this run** (proven by reverting shellQuote alone: app+server counts unchanged
or improved; the movement module is a separate file and cannot affect them).

**Known blocker (pre-existing, NOT introduced here):** `app.test.ts` (2) and
`server.test.ts` (1, load-dependent) remain red. Root cause: these tests spawn
**real detached processes** and assert on process liveness / platform-breaker state
— which is nondeterministic in the cloud sandbox (a task shows `missing-process` →
platform control reads `degraded`, and an expected active task isn't found). The
proper fix is test hermeticity: inject a mock `SpawnBackgroundProcess` /
`backgroundTaskIsProcessRunning` into `OperatorCliApp` (the runtime already supports
the latter; the app doesn't expose it yet) so no real process runs — the same class
of fix as the `configHome` hermeticity work in the 2026-06-22 run. Logged to ROADMAP.

**New idea:** a **generalization eval harness** built on `MovementModel.evaluate()`:
hold out one recorded trajectory, train on the rest, and report replay fidelity +
backoff rate on the held-out sequence — turning "does the model generalize?" into a
tracked scalar per run. Pairs naturally with a synthetic event-stream generator so
we can sweep fidelity vs. dataset size without any real OS input.

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
