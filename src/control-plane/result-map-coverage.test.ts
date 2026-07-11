import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * Map-coverage guard for the control-plane RPC surface.
 *
 * `ControlPlaneServer.handle()` dispatches on a big `switch (method)`. Every
 * mapped method has an entry in `ControlPlaneResultMap`, which is what gives
 * `handle()` / the client facade a *typed* `result` instead of `unknown`. When
 * a new `case "x.y":` is added without a matching map entry, the result
 * silently degrades to `unknown` and nothing catches it — past runs
 * rediscovered the drift by grepping each time.
 *
 * This test scrapes both sets from the source of truth (server.ts) and asserts
 * every dispatched method is EITHER mapped OR explicitly allow-listed as
 * intentionally-`unknown` below. Adding a new RPC method now forces a
 * deliberate choice: map it, or add it to `ALLOW_UNKNOWN`.
 */

const SERVER_SRC = readFileSync(
  fileURLToPath(new URL("./server.ts", import.meta.url)),
  "utf8",
);

/**
 * Methods that intentionally return `unknown` for now. Shrinking this list (by
 * moving entries into `ControlPlaneResultMap`) is the ongoing typecheck-debt
 * paydown. Keep it sorted so diffs stay readable.
 */
const ALLOW_UNKNOWN: readonly string[] = [
  "background.tasks.active",
  "background.tasks.cancel",
  "background.tasks.get",
  "background.tasks.list",
  "background.tasks.output",
  "background.tasks.recover",
  "background.tasks.recoverAll",
  "background.tasks.start",
  "background.tasks.state",
  "background.tasks.sync",
  "memory.recall",
  "messages.get",
  "monitors.active",
  "notifications.send",
  "plugins.activate",
  "plugins.get",
  "plugins.list",
  "plugins.register",
  "plugins.update",
  "push.subscriptions.create",
  "push.subscriptions.delete",
  "push.subscriptions.list",
  "push.test",
  "runs.active",
  "runs.events",
  "runs.update",
  "skills.candidates.list",
  "skills.candidates.review",
  "skills.executable.create",
  "skills.executable.list",
  "skills.executable.run",
  "skills.executable.runs",
  "skills.list",
  "skills.promote",
  "subagents.active",
  "subagents.get",
  "subagents.list",
  "subagents.register",
  "subagents.spawn",
  "subagents.update",
];

/** All `case "x.y":` labels dispatched by `handle()`. */
function dispatchedMethods(): Set<string> {
  return new Set([...SERVER_SRC.matchAll(/case "([^"]+)":/g)].map((m) => m[1]));
}

/** All quoted keys inside the `ControlPlaneResultMap` interface body. */
function mappedMethods(): Set<string> {
  const start = SERVER_SRC.indexOf("export interface ControlPlaneResultMap {");
  expect(start, "ControlPlaneResultMap interface should exist").toBeGreaterThan(-1);
  const end = SERVER_SRC.indexOf("\n}", start);
  const body = SERVER_SRC.slice(start, end);
  return new Set([...body.matchAll(/^\s*"([^"]+)":/gm)].map((m) => m[1]));
}

describe("control-plane RPC map coverage", () => {
  it("scrapes a non-trivial number of methods from both sources", () => {
    // Guards the regexes themselves against silent breakage (e.g. a refactor
    // that renames the interface would otherwise make every check vacuous).
    expect(dispatchedMethods().size).toBeGreaterThan(80);
    expect(mappedMethods().size).toBeGreaterThan(40);
  });

  it("maps or explicitly allow-lists every dispatched method", () => {
    const mapped = mappedMethods();
    const allow = new Set(ALLOW_UNKNOWN);
    const uncovered = [...dispatchedMethods()]
      .filter((m) => !mapped.has(m) && !allow.has(m))
      .sort();
    expect(
      uncovered,
      "these RPC methods return `unknown` — add a ControlPlaneResultMap entry " +
        "or, deliberately, an ALLOW_UNKNOWN entry",
    ).toEqual([]);
  });

  it("keeps ALLOW_UNKNOWN honest — no entry is already mapped", () => {
    const mapped = mappedMethods();
    const stale = ALLOW_UNKNOWN.filter((m) => mapped.has(m));
    expect(
      stale,
      "these are mapped now; remove them from ALLOW_UNKNOWN",
    ).toEqual([]);
  });

  it("keeps ALLOW_UNKNOWN honest — every entry is actually dispatched", () => {
    const dispatched = dispatchedMethods();
    const dead = ALLOW_UNKNOWN.filter((m) => !dispatched.has(m));
    expect(dead, "these allow-listed methods are no longer dispatched").toEqual(
      [],
    );
  });

  it("has no dead ControlPlaneResultMap entries", () => {
    const dispatched = dispatchedMethods();
    const dead = [...mappedMethods()].filter((m) => !dispatched.has(m)).sort();
    expect(
      dead,
      "these map entries reference methods no `case` dispatches",
    ).toEqual([]);
  });

  it("keeps ALLOW_UNKNOWN sorted and duplicate-free", () => {
    const sorted = [...ALLOW_UNKNOWN].sort();
    expect(ALLOW_UNKNOWN).toEqual(sorted);
    expect(new Set(ALLOW_UNKNOWN).size).toBe(ALLOW_UNKNOWN.length);
  });
});
