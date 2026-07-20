import {
  buildMovementDataset,
  type MovementDataset,
  type MovementSequence,
  type MovementToken,
} from "./movement-model.js";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud with no access to a real machine's mouse/keyboard
 * events, so the movement-learning pipeline is validated against seeded
 * synthetic streams instead. A base "motif" (a canonical sequence of movement
 * tokens) is emitted across many episodes, with small, seeded mutations —
 * dropped, duplicated, or swapped steps — so the corpus contains new-but-related
 * variations. That lets tests exercise capture → dataset → train → generalize
 * without any OS input while keeping results reproducible.
 */

/** Small deterministic PRNG (mulberry32) — reproducible synthetic corpora. */
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

export type SyntheticMovementOptions = {
  /** Canonical movement motif every episode is derived from. */
  motif: MovementSequence;
  /** How many episodes to synthesize. */
  episodes: number;
  /** Seed for reproducibility. */
  seed?: number;
  /** Per-step probability of a mutation (drop/duplicate/swap). Default 0.15. */
  mutationRate?: number;
  /** Extra tokens the mutator may substitute in. Defaults to the motif tokens. */
  alphabet?: MovementToken[];
};

function mutateSequence(
  motif: MovementSequence,
  rng: () => number,
  mutationRate: number,
  alphabet: MovementToken[],
): MovementSequence {
  const out: MovementSequence = [];
  for (let i = 0; i < motif.length; i += 1) {
    const token = motif[i];
    if (rng() < mutationRate) {
      const roll = rng();
      if (roll < 0.34) {
        // drop this step
        continue;
      }
      if (roll < 0.67) {
        // duplicate this step
        out.push(token, token);
        continue;
      }
      // substitute a related token
      const replacement = alphabet[Math.floor(rng() * alphabet.length)] ?? token;
      out.push(replacement);
      continue;
    }
    out.push(token);
  }
  // Never emit an empty episode — fall back to the motif.
  return out.length > 0 ? out : [...motif];
}

/** Generate a reproducible set of movement sequences from a motif. */
export function generateSyntheticMovementSequences(options: SyntheticMovementOptions): MovementSequence[] {
  const rng = mulberry32(options.seed ?? 1);
  const mutationRate = options.mutationRate ?? 0.15;
  const alphabet = options.alphabet ?? [...new Set(options.motif)];
  const episodes = Math.max(0, Math.floor(options.episodes));
  const sequences: MovementSequence[] = [];
  for (let i = 0; i < episodes; i += 1) {
    sequences.push(mutateSequence(options.motif, rng, mutationRate, alphabet));
  }
  return sequences;
}

/** Generate a synthetic movement dataset ready for training. */
export function generateSyntheticMovementDataset(options: SyntheticMovementOptions): MovementDataset {
  const sequences = generateSyntheticMovementSequences(options);
  const vocabulary = new Set<MovementToken>();
  for (const sequence of sequences) {
    for (const token of sequence) {
      vocabulary.add(token);
    }
  }
  return { version: 1, sequences, vocabulary: [...vocabulary].sort() };
}

/**
 * Render synthetic sequences into a replay manifest so the full
 * replay → dataset path can be exercised (each sequence becomes one
 * trajectory of `action` events on a synthetic timeline).
 */
export function synthesizeReplayManifest(sequences: MovementSequence[], sessionId = "synthetic-session"): ReplayManifest {
  const events: ReplayTimelineEvent[] = [];
  const trajectoryIds: string[] = [];
  let ts = 0;
  sequences.forEach((sequence, index) => {
    const trajectoryId = `synthetic-traj-${index}`;
    trajectoryIds.push(trajectoryId);
    for (const token of sequence) {
      const [prefix, ...rest] = token.split(":");
      const name = rest.join(":") || token;
      if (prefix === "obs") {
        events.push({ kind: "observation", ts, trajectoryId, source: name, summary: `synthetic ${name}` });
      } else {
        events.push({ kind: "action", ts, trajectoryId, tool: name, summary: `synthetic ${name}` });
      }
      ts += 1;
    }
  });
  return { version: 1, sessionId, trajectoryIds, eventCount: events.length, events };
}

/** Convenience: synthesize sequences → replay manifest → dataset in one call. */
export function syntheticDatasetViaReplay(options: SyntheticMovementOptions): MovementDataset {
  const sequences = generateSyntheticMovementSequences(options);
  return buildMovementDataset([synthesizeReplayManifest(sequences)]);
}
