/**
 * Deterministic synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud with no access to a real machine, so the movement
 * learning subsystem is validated against *simulated* event streams. This module
 * produces reproducible {@link ReplaySource} streams (seeded, no `Math.random`)
 * shaped like real captured movements — a UI "grammar" of actions with
 * interleaved observations — so the capture → dataset → train → infer → eval loop
 * can be exercised end-to-end in tests and generalization harnesses.
 */

import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { ReplaySource } from "./movement-model.js";

/** Tiny deterministic PRNG (mulberry32) so streams are reproducible from a seed. */
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

export type SyntheticMovementGrammar = {
  /** Ordered "motifs" — recurring movement phrases the generator strings together. */
  motifs: MovementMotif[];
};

export type MovementMotif = {
  tool: string;
  summary: string;
  /** Optional observation emitted just before the action. */
  observation?: { source: string; summary: string };
};

export type GenerateSyntheticStreamsOptions = {
  seed?: number;
  /** Number of independent sessions/streams to emit (default: 4). */
  streamCount?: number;
  /** Movements per stream (default: 8). */
  stepsPerStream?: number;
  /** Milliseconds between successive events (default: 1000). */
  stepIntervalMs?: number;
  /** Emit observation events alongside actions when a motif defines one (default: true). */
  includeObservations?: boolean;
  grammar?: SyntheticMovementGrammar;
};

/** A small default UI grammar: open → navigate → edit → save style motifs. */
export const DEFAULT_MOVEMENT_GRAMMAR: SyntheticMovementGrammar = {
  motifs: [
    { tool: "window.focus", summary: "focus editor window", observation: { source: "os", summary: "editor foreground" } },
    { tool: "pointer.move", summary: "move cursor to sidebar" },
    { tool: "pointer.click", summary: "click file entry", observation: { source: "ui", summary: "file opened" } },
    { tool: "keyboard.type", summary: "type search query" },
    { tool: "keyboard.shortcut", summary: "press cmd+s save", observation: { source: "ui", summary: "document saved" } },
    { tool: "pointer.scroll", summary: "scroll results list" },
    { tool: "pointer.click", summary: "click run button", observation: { source: "ui", summary: "task started" } },
  ],
};

/**
 * Generate reproducible synthetic replay streams. Each stream is a Markov-ish
 * walk over the grammar's motifs, so streams share structure (repeated motifs)
 * while differing in detail — ideal for measuring generalization.
 */
export function generateSyntheticMovementStreams(
  options: GenerateSyntheticStreamsOptions = {},
): ReplaySource[] {
  const seed = options.seed ?? 1;
  const streamCount = Math.max(1, options.streamCount ?? 4);
  const stepsPerStream = Math.max(1, options.stepsPerStream ?? 8);
  const stepIntervalMs = options.stepIntervalMs ?? 1000;
  const includeObservations = options.includeObservations ?? true;
  const grammar = options.grammar ?? DEFAULT_MOVEMENT_GRAMMAR;
  const motifs = grammar.motifs;
  if (motifs.length === 0) {
    return [];
  }

  const streams: ReplaySource[] = [];
  for (let s = 0; s < streamCount; s += 1) {
    const rng = mulberry32(seed + s * 7919);
    const events: ReplayTimelineEvent[] = [];
    let ts = 0;
    let motifIndex = Math.floor(rng() * motifs.length);
    for (let step = 0; step < stepsPerStream; step += 1) {
      const motif = motifs[motifIndex]!;
      if (includeObservations && motif.observation) {
        events.push({
          kind: "observation",
          ts,
          trajectoryId: `synthetic-${s}`,
          source: motif.observation.source,
          summary: motif.observation.summary,
        });
        ts += Math.max(1, Math.floor(stepIntervalMs / 2));
      }
      events.push({
        kind: "action",
        ts,
        trajectoryId: `synthetic-${s}`,
        tool: motif.tool,
        summary: motif.summary,
      });
      ts += stepIntervalMs;
      // Advance mostly forward through the grammar with occasional jumps,
      // producing overlapping-but-varied motif sequences across streams.
      const roll = rng();
      if (roll < 0.7) {
        motifIndex = (motifIndex + 1) % motifs.length;
      } else if (roll < 0.85) {
        motifIndex = (motifIndex + 2) % motifs.length;
      } else {
        motifIndex = Math.floor(rng() * motifs.length);
      }
    }
    streams.push({ id: `synthetic-session-${s}`, events });
  }

  return streams;
}
