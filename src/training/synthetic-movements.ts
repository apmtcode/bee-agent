import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud with NO access to a real machine, so the
 * capture→dataset→train→replay→generalize loop is validated against *simulated*
 * movement streams produced here. A stream is generated from a small movement
 * grammar (an ordered list of {tool} "verbs"), optionally perturbed to model
 * natural variation, and emitted as a {@link ReplayManifest} identical in shape
 * to what the real capture pipeline records — so nothing downstream can tell the
 * difference between simulated and captured data.
 *
 * Deterministic: sequences are driven by a seed and a pure PRNG, never
 * `Math.random`, so tests are reproducible and cloud-safe.
 */

export type MovementGrammar = {
  /** Ordered movement verbs, e.g. ["focus.window", "mouse.click", "keyboard.type"]. */
  verbs: string[];
  /** Optional observation sources interleaved before actions, e.g. ["screen"]. */
  observationSources?: string[];
};

export type SyntheticStreamOptions = {
  sessionId: string;
  grammar: MovementGrammar;
  seed?: number;
  /**
   * Per-step probability (0..1) of dropping a verb, modelling natural variation
   * so held-out streams are *related but not identical* to training streams.
   */
  dropoutRate?: number;
  /** Base timestamp (ms) for the first event; each event advances by `stepMs`. */
  startTs?: number;
  stepMs?: number;
};

/**
 * Build one synthetic {@link ReplayManifest} from a movement grammar. With
 * `dropoutRate` 0 the manifest reproduces the grammar exactly (use for the
 * "repeat recorded movements" path); with dropout > 0 it produces a related
 * variant (use for the "generalize" path).
 */
export function generateSyntheticReplay(options: SyntheticStreamOptions): ReplayManifest {
  const { sessionId, grammar } = options;
  const dropoutRate = clamp01(options.dropoutRate ?? 0);
  const stepMs = options.stepMs ?? 10;
  const random = makePrng(options.seed ?? 1);
  const observationSources = grammar.observationSources ?? [];

  const events: ReplayTimelineEvent[] = [];
  let ts = options.startTs ?? 0;

  for (let index = 0; index < grammar.verbs.length; index += 1) {
    const verb = grammar.verbs[index]!;
    if (dropoutRate > 0 && random() < dropoutRate) {
      continue;
    }
    const source = observationSources[index % Math.max(observationSources.length, 1)];
    if (source) {
      events.push({
        kind: "observation",
        ts,
        trajectoryId: `${sessionId}-traj`,
        source,
        summary: `observed ${source} before ${verb}`,
      });
      ts += stepMs;
    }
    events.push({
      kind: "action",
      ts,
      trajectoryId: `${sessionId}-traj`,
      tool: verb,
      summary: `perform ${verb}`,
    });
    ts += stepMs;
  }

  return {
    version: 1,
    sessionId,
    trajectoryIds: [`${sessionId}-traj`],
    eventCount: events.length,
    events,
  };
}

/**
 * Generate a corpus of `count` related streams from the same grammar, each with
 * a distinct seed so variants differ. Handy for building a training set plus a
 * held-out generalization set from one grammar.
 */
export function generateSyntheticCorpus(params: {
  grammar: MovementGrammar;
  count: number;
  seed?: number;
  dropoutRate?: number;
  sessionPrefix?: string;
}): ReplayManifest[] {
  const baseSeed = params.seed ?? 1;
  const prefix = params.sessionPrefix ?? "synthetic";
  const replays: ReplayManifest[] = [];
  for (let index = 0; index < params.count; index += 1) {
    replays.push(
      generateSyntheticReplay({
        sessionId: `${prefix}-${index}`,
        grammar: params.grammar,
        seed: baseSeed + index * 7919,
        dropoutRate: params.dropoutRate,
      }),
    );
  }
  return replays;
}

/** Small deterministic PRNG (mulberry32); avoids Math.random for reproducibility. */
function makePrng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}
