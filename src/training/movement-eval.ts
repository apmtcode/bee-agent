import type {
  MovementDataset,
  MovementEvent,
  MovementModelArtifact,
  MovementModelBackend,
  MovementSequence,
} from "./movement-model.js";
import { movementTokenKey } from "./movement-model.js";

/**
 * Generalization eval harness + deterministic synthetic event-stream generator
 * for the movement-learning subsystem. Because the engine runs in the cloud
 * with no access to real OS input, these synthetic streams stand in for real
 * recordings so the capture -> dataset -> train -> infer loop can be validated
 * end to end and regression-tested.
 */

/** Small deterministic PRNG (mulberry32) so synthetic corpora are reproducible. */
function makeRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(rng: () => number, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length) % items.length];
}

export type SyntheticCorpusOptions = {
  seed?: number;
  sequenceCount?: number;
  /** Apps the synthetic operator moves within. */
  apps?: readonly string[];
};

const DEFAULT_APPS = ["mail", "browser", "editor"] as const;

/**
 * Generate a corpus of related workflow movement sequences with genuine
 * structure the model can learn: each sequence focuses an app, performs a
 * variable run of scrolls, then a primary per-app action. The per-app primary
 * action is deterministic, so a model that learns app-conditioned structure
 * generalizes to held-out sequences far better than a majority-class baseline.
 */
export function generateSyntheticMovementCorpus(options: SyntheticCorpusOptions = {}): MovementDataset {
  const rng = makeRng(options.seed ?? 1);
  const apps = options.apps ?? DEFAULT_APPS;
  const sequenceCount = options.sequenceCount ?? 24;
  const primaryByApp: Record<string, string> = {
    mail: "compose",
    browser: "open-link",
    editor: "save",
  };

  const sequences: MovementSequence[] = [];
  for (let i = 0; i < sequenceCount; i += 1) {
    const app = pick(rng, apps);
    const events: MovementEvent[] = [{ context: app, channel: "os", action: "focus-changed", target: app }];
    const scrolls = 1 + Math.floor(rng() * 3);
    for (let s = 0; s < scrolls; s += 1) {
      const direction = rng() < 0.7 ? "down" : "up";
      events.push({ context: app, channel: "device", action: `scroll:${direction}`, target: "content" });
    }
    const primary = primaryByApp[app] ?? "activate";
    events.push({ context: app, channel: "device", action: "tap", target: primary });
    events.push({ context: app, channel: "tool", action: primary, target: `${app}:${primary}` });
    sequences.push({ id: `synthetic-${i}`, events });
  }
  return { sequences };
}

/** Deterministic train/test split by sequence index (every Nth held out). */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdoutEvery = 4,
): { train: MovementDataset; test: MovementDataset } {
  const train: MovementSequence[] = [];
  const test: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if (holdoutEvery > 0 && index % holdoutEvery === holdoutEvery - 1) {
      test.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return { train: { sequences: train }, test: { sequences: test } };
}

export type MovementFidelityReport = {
  /** Number of (context -> next) predictions scored. */
  predictions: number;
  /** Exact next-movement matches. */
  correct: number;
  /** correct / predictions (0..1). */
  accuracy: number;
  /** How many predictions each backoff level produced. */
  levelCounts: Record<string, number>;
  /** Majority-class baseline accuracy over the same predictions (floor). */
  baselineAccuracy: number;
};

/**
 * Measure next-movement replay fidelity on held-out sequences: for each
 * position, feed the true prefix as context and check whether the predicted
 * next movement matches the recorded one. Reports the model accuracy, a
 * per-backoff-level breakdown, and the majority-class baseline so callers can
 * assert the model genuinely generalizes rather than guessing the mode.
 */
export function evaluateMovementFidelity(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  testSequences: MovementSequence[],
): MovementFidelityReport {
  const majority = majorityToken(model.priorCounts);
  let predictions = 0;
  let correct = 0;
  let baselineCorrect = 0;
  const levelCounts: Record<string, number> = {};

  for (const sequence of testSequences) {
    for (let i = 1; i < sequence.events.length; i += 1) {
      const context = sequence.events.slice(0, i);
      const actual = movementTokenKey(sequence.events[i]);
      const prediction = backend.predict(model, context);
      predictions += 1;
      levelCounts[prediction.level] = (levelCounts[prediction.level] ?? 0) + 1;
      if (prediction.event && movementTokenKey(prediction.event) === actual) {
        correct += 1;
      }
      if (majority !== undefined && majority === actual) {
        baselineCorrect += 1;
      }
    }
  }

  return {
    predictions,
    correct,
    accuracy: predictions > 0 ? correct / predictions : 0,
    levelCounts,
    baselineAccuracy: predictions > 0 ? baselineCorrect / predictions : 0,
  };
}

function majorityToken(priorCounts: Record<string, number>): string | undefined {
  let bestKey: string | undefined;
  let bestCount = -1;
  for (const key of Object.keys(priorCounts).sort()) {
    const count = priorCounts[key] ?? 0;
    if (count > bestCount) {
      bestCount = count;
      bestKey = key;
    }
  }
  return bestKey;
}
