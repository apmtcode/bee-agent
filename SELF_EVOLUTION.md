# bee-agent self-evolution log

This file is the running memory of the autonomous self-evolution engine. Each
run appends a dated entry: what was audited, what changed, test results, and at
least one new idea. Newest entries first.

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
