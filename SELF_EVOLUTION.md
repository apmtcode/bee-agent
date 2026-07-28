# bee-agent self-evolution log

This file is the running memory of the autonomous self-evolution engine. Each
run appends a dated entry: what was audited, what changed, test results, and at
least one new idea. Newest entries first.

---

## 2026-07-28 (run 10) — 🧠 Pluggable local movement-model backend + generalization eval (objective #2c/#2d)

**Audited:** The local-movement learning subsystem (standing objective #2). Prior
runs built the *plumbing* — capture (`src/capture/*`), reviewed export, and a
training **runner** (`src/training/runner.ts`). But that runner only emits a
shell plan that shells out to `mlx_lm`/`axolotl` on Apple Silicon: there was **no
in-process model at all**, so objective 2(c) "post-train a local model … to
repeat the recorded movements" and 2(d) "generalize to … new but related
movements" were completely unrealized and **untestable in the cloud**. That —
not more typecheck paydown (125, all test-only) — was the highest-value gap.

**Changed (additive, one new module + tests):** added
`src/training/movement-model.ts` — a **pluggable** local movement-model
subsystem behind a `MovementModelBackend` interface (train / predictNext /
generate), with a fully deterministic in-process default backend
`MarkovMovementBackend`:
- **Tokenizer/dataset:** `parseMovementToken` turns an action summary
  (`"tapped Send button"`) into a structured `{tool,verb,target}` token;
  `tokenizeReplayManifest` / `buildMovementDataset` /
  `buildMovementDatasetFromTrajectories` build a `MovementDataset` straight from
  existing `ReplayManifest`s and `TrajectorySpan`s — reusing the capture schema,
  no new capture surface.
- **Train (2c):** katz-style **back-off** Markov model — counts next-token
  frequencies for every context length `0..order` (order 2 default). Serializes
  to a plain-JSON `MovementModel` (per-order transition tables + vocabulary).
- **Repeat (2c):** `predictNext`/`generate` regenerate a recorded sequence from
  a prime, deterministic argmax with sorted-key tie-break, with a cycle guard so
  the walk can't loop forever.
- **Generalize (2d):** unseen contexts **back off** to shorter contexts / the
  global prior — the mechanism that recovers next-movements from shared task
  structure the model never saw verbatim.
- **Pluggability:** `createMovementModelBackend(kind)` registry — `"markov"`
  today; documented seam for a real on-device small model behind the same
  interface (cloud has no device access, so the shipped backend is native-dep-free).
- **Synthetic generator + eval harness:** `synthesizeMovementSequences` emits
  deterministic *related-but-distinct* task variants (no RNG — cloud-safe);
  `partitionSequences` does a leak-free train/held-out split;
  `evaluateReplayFidelity` measures held-out next-movement accuracy with a
  per-backoff-order breakdown. Ships **three** queued roadmap items at once
  (pluggable backend + mock, synthetic generator, generalization eval).

Exported all of it from `src/index.ts` (barrel). No existing file's behavior
changed.

**Test results:** new `movement-model.test.ts` — **17 tests** covering parse,
tokenize, train determinism, exact repeat-from-prime, backoff generalization,
cycle-stop, and a held-out generalization eval asserting **>0.5 accuracy on
sequences never trained on**. `npm run build` ✅. `npm run typecheck:src` ✅
(exit 0 — source stays clean). Full suite **174 → 191/191**, green on **2/2**
consecutive runs (flake sentinel). Full `tsc` **125** (unchanged — no
regression; the new source module is clean, remaining 125 are all test-only).

**New idea:** a **`MovementReplayEngine` bridge** that feeds a generated
`MovementToken[]` back through the existing `ReplayRuntimeService`/device-adapter
in *dry-run* mode — closing the loop capture→train→**infer→act(sim)** and letting
the eval harness score *executed* fidelity (did the simulated actor reach the
recorded end-state?) rather than only next-token accuracy. Pairs naturally with a
second backend (`"frequency-prior"` or a tiny embedding-nearest-neighbour) so the
eval harness can **rank backends** on the same held-out set — turning "pluggable"
into a measurable bake-off instead of a single implementation.

---

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
