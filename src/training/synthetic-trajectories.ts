import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud with no access to a real mouse/keyboard, so the
 * movement-learning pipeline must be validated against *simulated* event
 * streams. This module produces reproducible {@link TrajectorySpan}s from small
 * movement "motifs" (open -> focus -> click -> type -> save, etc.), with a
 * seeded RNG so the same seed always yields the same dataset. Related-but-novel
 * variants let the generalization eval measure backoff quality on held-out
 * trajectories.
 */

export type MovementStep =
  | { kind: "observation"; source: string }
  | { kind: "action"; tool: string; summary: string };

export type SyntheticMovementScenario = {
  name: string;
  steps: MovementStep[];
};

export type GenerateSyntheticTrajectoriesOptions = {
  scenarios: SyntheticMovementScenario[];
  /** How many spans to emit per scenario. Default 4. */
  spansPerScenario?: number;
  /** Seed for the deterministic RNG. Default 1. */
  seed?: number;
  /** Base timestamp (ms) for the first event. Default 0. */
  startTs?: number;
  /** Session id assigned to every generated span. Default "synthetic". */
  sessionId?: string;
};

/** A seeded, dependency-free PRNG (mulberry32) — deterministic across runs. */
export function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generate deterministic trajectory spans from movement scenarios.
 *
 * Timestamps advance monotonically with small seeded jitter so the merged
 * timeline still orders events by their scenario position, exercising the same
 * sort path as real captures.
 */
export function generateSyntheticTrajectories(
  options: GenerateSyntheticTrajectoriesOptions,
): TrajectorySpan[] {
  const spansPerScenario = options.spansPerScenario ?? 4;
  const rng = createSeededRng(options.seed ?? 1);
  const sessionId = options.sessionId ?? "synthetic";
  let ts = options.startTs ?? 0;
  const spans: TrajectorySpan[] = [];

  for (const scenario of options.scenarios) {
    for (let index = 0; index < spansPerScenario; index += 1) {
      const observations: TrajectorySpan["observations"] = [];
      const actions: TrajectorySpan["actions"] = [];
      for (const step of scenario.steps) {
        ts += 1 + Math.floor(rng() * 3);
        if (step.kind === "observation") {
          observations.push({
            kind: "observation",
            source: step.source,
            summary: `${step.source} observed`,
            ts,
          });
        } else {
          actions.push({
            kind: "action",
            tool: step.tool,
            summary: step.summary,
            ts,
          });
        }
      }
      spans.push(
        buildTrajectorySpan({
          id: `${scenario.name}-${index}`,
          sessionId,
          captureTier: "full",
          observations,
          actions,
          outcome: { status: "success", summary: `${scenario.name} completed` },
        }),
      );
    }
  }

  return spans;
}

/**
 * A family of related movement scenarios: a shared training motif plus
 * held-out variants that reuse its prefix but diverge at the tail. The eval
 * harness trains on `train` and measures prediction quality on `heldOut`,
 * where correct predictions on the divergent tail can only come from context
 * backoff — i.e. genuine generalization, not memorization.
 */
export type MovementScenarioFamily = {
  train: SyntheticMovementScenario[];
  heldOut: SyntheticMovementScenario[];
};

/** A canonical desktop-automation scenario family for tests and demos. */
export function desktopMovementFamily(): MovementScenarioFamily {
  const openFocusClick: MovementStep[] = [
    { kind: "observation", source: "window" },
    { kind: "action", tool: "app.open", summary: "open editor" },
    { kind: "observation", source: "focus" },
    { kind: "action", tool: "pointer.click", summary: "click document" },
  ];

  return {
    train: [
      {
        name: "edit-and-save",
        steps: [
          ...openFocusClick,
          { kind: "action", tool: "keyboard.type", summary: "type body" },
          { kind: "action", tool: "keyboard.shortcut", summary: "save (cmd+s)" },
        ],
      },
      {
        name: "edit-and-close",
        steps: [
          ...openFocusClick,
          { kind: "action", tool: "keyboard.type", summary: "type body" },
          { kind: "action", tool: "window.close", summary: "close window" },
        ],
      },
    ],
    heldOut: [
      {
        // Shares the open->focus->click prefix; the tail (find) is novel, so a
        // correct next-action prediction after the prefix must come from backoff.
        name: "edit-and-find",
        steps: [
          ...openFocusClick,
          { kind: "action", tool: "keyboard.type", summary: "type body" },
          { kind: "action", tool: "keyboard.shortcut", summary: "find (cmd+f)" },
        ],
      },
    ],
  };
}
