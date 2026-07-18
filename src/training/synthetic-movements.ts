import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import type { MovementSequence } from "./movement-model.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud with no access to a real machine's mouse /
 * keyboard / window events, so the movement-learning subsystem must be
 * validated against *simulated* input. This produces reproducible gesture
 * trajectories from a small library of "motifs" (short, structured movement
 * phrases) with seedable variation, so tests can train a model, hold out
 * sequences, and measure generalization without any OS involvement.
 *
 * Determinism: a tiny LCG seeded by an integer — no `Math.random`, no `Date` —
 * so the same seed always yields the same stream (a hard requirement in this
 * environment, where those globals are unavailable to keep runs reproducible).
 */

export type SyntheticGesture = {
  tool: string;
  gesture: string;
  qualifier?: string;
};

/** A named movement phrase the generator can emit, optionally with variation. */
export type MovementMotif = {
  name: string;
  /** The core, always-present steps of the phrase. */
  steps: SyntheticGesture[];
  /** Optional steps appended with the given probability (deterministic per seed). */
  optionalTail?: { steps: SyntheticGesture[]; probability: number };
};

export type SyntheticStreamOptions = {
  seed?: number;
  /** How many trajectories to emit. Default 12. */
  count?: number;
  /** Motif library to sample from. Defaults to {@link DEFAULT_MOVEMENT_MOTIFS}. */
  motifs?: MovementMotif[];
  /** Base epoch millis for the first event; each step advances deterministically. */
  startTs?: number;
  sessionId?: string;
};

/** A compact deterministic PRNG (Numerical Recipes LCG). */
class Lcg {
  private state: number;
  constructor(seed: number) {
    // Keep the seed in a stable 32-bit range so results are portable.
    this.state = (Math.floor(seed) % 2147483647) || 1;
    if (this.state < 0) {
      this.state += 2147483646;
    }
  }
  /** Next float in [0, 1). */
  next(): number {
    this.state = (this.state * 1103515245 + 12345) & 0x7fffffff;
    return this.state / 0x7fffffff;
  }
  /** Next integer in [0, bound). */
  int(bound: number): number {
    return bound <= 0 ? 0 : Math.floor(this.next() * bound);
  }
}

/**
 * A default motif library resembling common desktop workflows: open-and-search,
 * navigate-and-select, drag-reorder, form-fill. The motifs deliberately share
 * sub-phrases (e.g. several end in a confirm shortcut) so a trained model has
 * real structure to generalize across.
 */
export const DEFAULT_MOVEMENT_MOTIFS: MovementMotif[] = [
  {
    name: "open-and-search",
    steps: [
      { tool: "device", gesture: "shortcut", qualifier: "cmd-space" },
      { tool: "device", gesture: "type", qualifier: "query" },
      { tool: "device", gesture: "tap", qualifier: "result" },
    ],
    optionalTail: { steps: [{ tool: "device", gesture: "shortcut", qualifier: "enter" }], probability: 0.5 },
  },
  {
    name: "navigate-and-select",
    steps: [
      { tool: "device", gesture: "scroll", qualifier: "down" },
      { tool: "device", gesture: "scroll", qualifier: "down" },
      { tool: "device", gesture: "tap", qualifier: "item" },
      { tool: "device", gesture: "shortcut", qualifier: "enter" },
    ],
  },
  {
    name: "drag-reorder",
    steps: [
      { tool: "device", gesture: "tap", qualifier: "handle" },
      { tool: "device", gesture: "swipe", qualifier: "up" },
      { tool: "device", gesture: "tap", qualifier: "drop" },
    ],
    optionalTail: { steps: [{ tool: "device", gesture: "shortcut", qualifier: "save" }], probability: 0.6 },
  },
  {
    name: "form-fill",
    steps: [
      { tool: "device", gesture: "tap", qualifier: "field" },
      { tool: "device", gesture: "type", qualifier: "value" },
      { tool: "device", gesture: "shortcut", qualifier: "tab" },
      { tool: "device", gesture: "type", qualifier: "value" },
      { tool: "device", gesture: "shortcut", qualifier: "save" },
    ],
  },
];

function gestureToAction(gesture: SyntheticGesture, ts: number): TrajectoryAction {
  return {
    kind: "action",
    tool: gesture.tool,
    summary: gesture.qualifier ? `${gesture.gesture} ${gesture.qualifier}` : gesture.gesture,
    ts,
    metadata: {
      gesture: gesture.gesture,
      ...(gesture.qualifier ? { direction: gesture.qualifier } : {}),
    },
  };
}

/**
 * Generate a deterministic list of synthetic movement trajectories. Same seed +
 * options ⇒ identical output.
 */
export function generateSyntheticTrajectories(options: SyntheticStreamOptions = {}): TrajectorySpan[] {
  const motifs = options.motifs ?? DEFAULT_MOVEMENT_MOTIFS;
  if (motifs.length === 0) {
    return [];
  }
  const count = Math.max(0, Math.floor(options.count ?? 12));
  const rng = new Lcg(options.seed ?? 1);
  const sessionId = options.sessionId ?? "synthetic-session";
  let ts = options.startTs ?? 1_700_000_000_000;

  const spans: TrajectorySpan[] = [];
  for (let i = 0; i < count; i += 1) {
    const motif = motifs[rng.int(motifs.length)]!;
    const steps: SyntheticGesture[] = [...motif.steps];
    if (motif.optionalTail && rng.next() < motif.optionalTail.probability) {
      steps.push(...motif.optionalTail.steps);
    }
    const actions = steps.map((step) => {
      ts += 1000 + rng.int(500);
      return gestureToAction(step, ts);
    });
    spans.push(
      buildTrajectorySpan({
        id: `synthetic-${i}`,
        sessionId,
        captureTier: "app",
        actions,
        outcome: { status: "success", summary: `${motif.name} completed` },
      }),
    );
  }
  return spans;
}

/** Convenience: synthetic trajectories already tokenized into sequences. */
export function generateSyntheticSequences(options: SyntheticStreamOptions = {}): MovementSequence[] {
  // Local import avoided to keep the module dependency-light; tokenize inline
  // via the same rule the model uses so callers get train-ready sequences.
  return generateSyntheticTrajectories(options).map((span) =>
    [...span.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => {
        const metadata = action.metadata ?? {};
        const gesture = typeof metadata.gesture === "string" ? slug(metadata.gesture) : "";
        const qualifier = typeof metadata.direction === "string" ? slug(metadata.direction) : "";
        const tool = slug(action.tool) || "action";
        if (gesture) {
          return qualifier ? `${tool}:${gesture}:${qualifier}` : `${tool}:${gesture}`;
        }
        return tool;
      }),
  );
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}
