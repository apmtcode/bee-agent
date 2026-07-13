/**
 * Synthetic movement-stream generator.
 *
 * The engine has no access to the user's real machine, so recorded input is
 * unavailable in cloud/CI. This deterministic generator fabricates realistic
 * replay timelines from a small library of UI "flows" (login, compose-email,
 * file-save, ...) so the capture -> dataset -> train -> replay pipeline can be
 * validated end-to-end without real OS input. Determinism (seedable LCG) keeps
 * tests reproducible.
 */
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

export type SyntheticFlowStep =
  | { kind: "observation"; source: string; summary: string }
  | { kind: "action"; tool: string; summary: string };

export type SyntheticFlow = {
  name: string;
  steps: SyntheticFlowStep[];
};

/** A small library of related UI flows. Related flows share tokens on purpose. */
export const DEFAULT_SYNTHETIC_FLOWS: SyntheticFlow[] = [
  {
    name: "login",
    steps: [
      { kind: "observation", source: "os", summary: "focused login window" },
      { kind: "action", tool: "device", summary: "tapped username field" },
      { kind: "action", tool: "device", summary: "typed username" },
      { kind: "action", tool: "device", summary: "tapped password field" },
      { kind: "action", tool: "device", summary: "typed password" },
      { kind: "action", tool: "device", summary: "tapped sign in" },
      { kind: "observation", source: "os", summary: "opened dashboard" },
    ],
  },
  {
    name: "compose-email",
    steps: [
      { kind: "observation", source: "os", summary: "focused mail window" },
      { kind: "action", tool: "device", summary: "tapped compose button" },
      { kind: "action", tool: "device", summary: "typed subject" },
      { kind: "action", tool: "device", summary: "typed body" },
      { kind: "action", tool: "device", summary: "tapped send" },
      { kind: "observation", source: "os", summary: "sent confirmation" },
    ],
  },
  {
    name: "save-file",
    steps: [
      { kind: "observation", source: "os", summary: "focused editor window" },
      { kind: "action", tool: "device", summary: "tapped save button" },
      { kind: "action", tool: "device", summary: "typed file name" },
      { kind: "action", tool: "device", summary: "tapped confirm" },
      { kind: "observation", source: "os", summary: "saved file" },
    ],
  },
];

/** Small seedable PRNG (mulberry32) so generated streams are reproducible. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SyntheticStreamOptions = {
  seed?: number;
  /** Number of flow instances to emit. */
  count: number;
  flows?: SyntheticFlow[];
  /** Milliseconds between consecutive events (default 250). */
  stepIntervalMs?: number;
  /** Base timestamp for the first event (default 0 — kept deterministic). */
  startTs?: number;
};

/**
 * Emit `count` replay manifests, each one instance of a randomly-chosen flow.
 * Each manifest looks exactly like {@link buildReplayManifest} output so it can
 * feed the tokenizer/dataset builders unchanged.
 */
export function generateSyntheticReplayStream(options: SyntheticStreamOptions): ReplayManifest[] {
  const flows = options.flows ?? DEFAULT_SYNTHETIC_FLOWS;
  if (flows.length === 0) {
    return [];
  }
  const random = mulberry32(options.seed ?? 1);
  const interval = options.stepIntervalMs ?? 250;
  const startTs = options.startTs ?? 0;
  const manifests: ReplayManifest[] = [];

  for (let instance = 0; instance < options.count; instance += 1) {
    const flow = flows[Math.floor(random() * flows.length)] ?? flows[0]!;
    const trajectoryId = `synthetic-${flow.name}-${instance}`;
    const sessionId = `synthetic-session-${instance}`;
    let ts = startTs + instance * flow.steps.length * interval;
    const events: ReplayTimelineEvent[] = flow.steps.map((step) => {
      const eventTs = ts;
      ts += interval;
      if (step.kind === "observation") {
        return { kind: "observation", ts: eventTs, trajectoryId, source: step.source, summary: step.summary };
      }
      return { kind: "action", ts: eventTs, trajectoryId, tool: step.tool, summary: step.summary };
    });
    manifests.push({
      version: 1,
      sessionId,
      trajectoryIds: [trajectoryId],
      eventCount: events.length,
      events,
    });
  }

  return manifests;
}
