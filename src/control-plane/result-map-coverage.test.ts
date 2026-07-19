import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * Guard: every method dispatched by `OperatorControlPlaneServer.handle`'s
 * switch must have a typed entry in `ControlPlaneResultMap` (or be explicitly
 * allow-listed below as intentionally `unknown`). Without this, a newly added
 * RPC method silently resolves to `unknown` at every call site — the exact debt
 * runs 5–8 spent paying down. This test turns that regression into a failure at
 * authoring time.
 *
 * The map is a compile-time-only TypeScript interface, so both the switch cases
 * and the map keys are scraped from the server source text.
 */

const serverSource = readFileSync(
  fileURLToPath(new URL("./server.ts", import.meta.url)),
  "utf8",
);

/** RPC method names are dotted (`family.action`); status/kind cases are not. */
const METHOD_PATTERN = /[a-z][a-zA-Z]*(?:\.[a-zA-Z]+)+/;

/**
 * Methods deliberately left unmapped (resolve to `unknown`). Keep this empty —
 * every entry here is type-safety debt. If a method genuinely cannot be typed,
 * add it with a comment explaining why.
 */
const INTENTIONALLY_UNKNOWN = new Set<string>([]);

function extractSwitchCases(source: string): Set<string> {
  const cases = new Set<string>();
  const caseRegex = new RegExp(`case "(${METHOD_PATTERN.source})":`, "g");
  for (const match of source.matchAll(caseRegex)) {
    cases.add(match[1]!);
  }
  return cases;
}

function extractMapKeys(source: string): Set<string> {
  const anchor = "export interface ControlPlaneResultMap";
  const anchorIndex = source.indexOf(anchor);
  if (anchorIndex === -1) {
    throw new Error("could not locate ControlPlaneResultMap in server source");
  }
  const openBrace = source.indexOf("{", anchorIndex);
  let depth = 0;
  let end = openBrace;
  for (let i = openBrace; i < source.length; i += 1) {
    const char = source[i];
    if (char === "{") depth += 1;
    else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        end = i;
        break;
      }
    }
  }
  const body = source.slice(openBrace + 1, end);
  const keys = new Set<string>();
  const keyRegex = new RegExp(`"(${METHOD_PATTERN.source})"\\s*:`, "g");
  for (const match of body.matchAll(keyRegex)) {
    keys.add(match[1]!);
  }
  return keys;
}

describe("ControlPlaneResultMap coverage", () => {
  const cases = extractSwitchCases(serverSource);
  const mapped = extractMapKeys(serverSource);

  it("scrapes a plausible number of methods from the source", () => {
    // Sanity-check the regexes still match the real file shape.
    expect(cases.size).toBeGreaterThan(80);
    expect(mapped.size).toBeGreaterThan(80);
  });

  it("maps (or explicitly allow-lists) every dispatched RPC method", () => {
    const unmapped = [...cases]
      .filter((method) => !mapped.has(method) && !INTENTIONALLY_UNKNOWN.has(method))
      .sort();
    expect(unmapped, `unmapped RPC methods missing a ControlPlaneResultMap entry: ${unmapped.join(", ")}`).toEqual([]);
  });

  it("has no stale allow-list entries", () => {
    const stale = [...INTENTIONALLY_UNKNOWN].filter(
      (method) => !cases.has(method) || mapped.has(method),
    );
    expect(stale, `allow-list entries that are no longer unmapped cases: ${stale.join(", ")}`).toEqual([]);
  });
});
