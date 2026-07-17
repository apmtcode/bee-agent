/**
 * Synthetic movement-stream generation + replay tokenization for the
 * movement-policy subsystem (standing objective #2b/#2d).
 *
 * The cloud engine has no access to a real mouse/keyboard/window stream, so we
 * validate the capture -> dataset -> train -> replay pipeline against *simulated*
 * movement streams. Sequences are stitched from a small library of movement
 * "motifs" (click, drag, type, scroll, ...) using a seeded PRNG, so:
 *
 *   - runs are fully deterministic (reproducible in CI), and
 *   - held-out sequences share motif structure with the training set, which is
 *     exactly what makes generalization measurable rather than luck.
 */

import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { MovementDataset, MovementSequence, MovementToken } from "./movement-policy.js";

/** Small, deterministic PRNG (mulberry32) -- no Math.random, no time deps. */
function createRng(seed: number): () => number {
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
 * Motif library: named short runs of device-agnostic movement tokens. Real
 * capture adapters emit these primitives (pointer/key/scroll); the motifs give
 * synthetic streams realistic local structure for the model to learn.
 */
export const DEFAULT_MOVEMENT_MOTIFS: Record<string, MovementToken[]> = {
  click: ["pointer.move", "pointer.down", "pointer.up"],
  doubleClick: ["pointer.move", "pointer.down", "pointer.up", "pointer.down", "pointer.up"],
  drag: ["pointer.down", "pointer.move", "pointer.move", "pointer.up"],
  type: ["key.down", "key.up", "key.down", "key.up"],
  scroll: ["scroll.begin", "scroll.delta", "scroll.delta", "scroll.end"],
  hover: ["pointer.move", "pointer.move"],
  focus: ["window.focus", "pointer.move"],
};

export type SyntheticMovementOptions = {
  seed: number;
  /** How many sequences to generate. */
  sequenceCount: number;
  /** Minimum / maximum motifs stitched per sequence (default 2..5). */
  minMotifs?: number;
  maxMotifs?: number;
  /** Motif library to draw from (defaults to {@link DEFAULT_MOVEMENT_MOTIFS}). */
  motifs?: Record<string, MovementToken[]>;
  /** Prefix for generated sequence ids (default "synthetic"). */
  idPrefix?: string;
};

/**
 * Generate a deterministic synthetic movement dataset. Same `seed` + options
 * always yields byte-identical sequences.
 */
export function generateSyntheticMovementDataset(options: SyntheticMovementOptions): MovementDataset {
  const motifs = options.motifs ?? DEFAULT_MOVEMENT_MOTIFS;
  const motifNames = Object.keys(motifs).sort();
  if (motifNames.length === 0) {
    return { sequences: [] };
  }
  const minMotifs = Math.max(1, Math.floor(options.minMotifs ?? 2));
  const maxMotifs = Math.max(minMotifs, Math.floor(options.maxMotifs ?? 5));
  const idPrefix = options.idPrefix ?? "synthetic";
  const rng = createRng(options.seed);

  const sequences: MovementSequence[] = [];
  for (let s = 0; s < options.sequenceCount; s += 1) {
    const motifCount = minMotifs + Math.floor(rng() * (maxMotifs - minMotifs + 1));
    const tokens: MovementToken[] = [];
    for (let m = 0; m < motifCount; m += 1) {
      const name = motifNames[Math.floor(rng() * motifNames.length)] ?? motifNames[0]!;
      const motif = motifs[name] ?? [];
      tokens.push(...motif);
    }
    sequences.push({ id: `${idPrefix}-${s}`, tokens });
  }
  return { sequences };
}

function isActionEvent(
  event: ReplayTimelineEvent,
): event is Extract<ReplayTimelineEvent, { kind: "action" }> {
  return event.kind === "action";
}

/**
 * Tokenize a replay action into a movement token. Defaults to the tool name
 * (the movement primitive); the recorded summary carries free-form detail we
 * deliberately drop to keep the vocabulary tight and learnable.
 */
export function movementTokenFromAction(tool: string, _summary: string): MovementToken {
  return tool;
}

/**
 * Derive movement sequences from reviewed replay manifests, so a real recorded
 * dataset (post-consent, post-review) can be fed to a {@link MovementPolicyBackend}
 * with the same shape the synthetic generator produces.
 */
export function movementSequencesFromReplays(
  replays: readonly ReplayManifest[],
): MovementSequence[] {
  return replays
    .map((replay) => ({
      id: replay.sessionId,
      tokens: replay.events
        .filter(isActionEvent)
        .map((event) => movementTokenFromAction(event.tool, event.summary)),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
}
