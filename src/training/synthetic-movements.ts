import { buildTrajectorySpan, type TrajectoryAction, type TrajectoryObservation, type TrajectorySpan } from "../capture/trajectory.js";
import type { MovementDataset, MovementSequence, MovementToken } from "./movement-dataset.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * bee-agent has no access to the user's real machine in the cloud, so the
 * capture → dataset → train → replay → generalize pipeline is validated against
 * simulated event streams produced here. Generation is fully deterministic
 * (seeded PRNG, no Math.random), so datasets are reproducible across runs and
 * tests.
 *
 * Each "family" is a task template with a mostly-fixed movement structure and a
 * few variable slots drawn from small pools. Training on some variants and
 * evaluating on held-out variants of the same families is what exercises
 * generalization: the model must reproduce the shared structure and back off on
 * the varying slots.
 */

// Deterministic 32-bit PRNG (mulberry32). Avoids Math.random for reproducibility.
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

function pick<T>(rng: () => number, pool: readonly T[]): T {
  return pool[Math.floor(rng() * pool.length) % pool.length]!;
}

type MovementFamily = {
  label: string;
  build: (rng: () => number) => MovementToken[];
};

const MOVEMENT_FAMILIES: readonly MovementFamily[] = [
  {
    label: "compose-message",
    build: (rng) => [
      { modality: "window", verb: "open", target: pick(rng, ["mail", "chat", "messages"]) },
      { modality: "pointer", verb: "tap", target: "compose" },
      { modality: "keyboard", verb: "type", target: "recipient" },
      { modality: "keyboard", verb: "type", target: pick(rng, ["body-a", "body-b", "body-c"]) },
      { modality: "pointer", verb: "tap", target: "send" },
    ],
  },
  {
    label: "file-search",
    build: (rng) => [
      { modality: "window", verb: "focus", target: "finder" },
      { modality: "keyboard", verb: "shortcut", target: "search" },
      { modality: "keyboard", verb: "type", target: pick(rng, ["query-a", "query-b", "query-c"]) },
      { modality: "pointer", verb: "scroll", target: "*", detail: pick(rng, ["down", "up"]) },
      { modality: "pointer", verb: "tap", target: "open" },
    ],
  },
  {
    label: "terminal-deploy",
    build: (rng) => [
      { modality: "window", verb: "focus", target: "terminal" },
      { modality: "command", verb: "run", target: "git-pull" },
      { modality: "command", verb: "run", target: pick(rng, ["build", "test-build"]) },
      { modality: "command", verb: "run", target: "deploy" },
    ],
  },
];

export type SyntheticMovementOptions = {
  seed?: number;
  /** Variants generated per family. */
  variantsPerFamily?: number;
  /** Restrict to specific family labels (default: all). */
  families?: string[];
};

function selectedFamilies(labels: string[] | undefined): readonly MovementFamily[] {
  if (!labels || labels.length === 0) {
    return MOVEMENT_FAMILIES;
  }
  const allow = new Set(labels);
  return MOVEMENT_FAMILIES.filter((family) => allow.has(family.label));
}

/** Generate labeled movement sequences directly (train / eval unit). */
export function synthesizeMovementSequences(options: SyntheticMovementOptions = {}): MovementSequence[] {
  const seed = options.seed ?? 1;
  const variants = Math.max(1, options.variantsPerFamily ?? 6);
  const families = selectedFamilies(options.families);
  const sequences: MovementSequence[] = [];
  let counter = 0;
  for (const family of families) {
    for (let variant = 0; variant < variants; variant += 1) {
      counter += 1;
      // Distinct, deterministic stream per (family, variant, seed).
      const rng = mulberry32(seed * 1_000_003 + counter * 97 + family.label.length);
      sequences.push({
        id: `${family.label}-${seed}-${variant}`,
        label: family.label,
        tokens: family.build(rng),
      });
    }
  }
  return sequences;
}

export function synthesizeMovementDataset(options: SyntheticMovementOptions = {}): MovementDataset {
  return { version: 1, sequences: synthesizeMovementSequences(options) };
}

/**
 * Emit a {@link TrajectorySpan} whose captured observations/actions tokenize
 * back to `sequence.tokens` — the inverse of `tokenizeTrajectory`. Used to
 * validate the full capture ↔ dataset round-trip without real OS input.
 */
export function sequenceToTrajectory(
  sequence: MovementSequence,
  params: { sessionId?: string; startTs?: number } = {},
): TrajectorySpan {
  const sessionId = params.sessionId ?? `sess-${sequence.id}`;
  const startTs = params.startTs ?? 1;
  const observations: TrajectoryObservation[] = [];
  const actions: TrajectoryAction[] = [];

  sequence.tokens.forEach((token, index) => {
    const ts = startTs + index;
    const emitted = emitCaptureEvent(token, ts);
    if (emitted.kind === "observation") {
      observations.push(emitted.observation);
    } else {
      actions.push(emitted.action);
    }
  });

  return buildTrajectorySpan({
    id: `traj-${sequence.id}`,
    sessionId,
    captureTier: "full",
    observations,
    actions,
    outcome: { status: "success", summary: sequence.label ?? "synthetic movement" },
  });
}

/** Generate synthetic trajectory spans (capture-level round-trip fixtures). */
export function synthesizeMovementTrajectories(options: SyntheticMovementOptions = {}): TrajectorySpan[] {
  return synthesizeMovementSequences(options).map((sequence) => sequenceToTrajectory(sequence));
}

type EmittedEvent =
  | { kind: "observation"; observation: TrajectoryObservation }
  | { kind: "action"; action: TrajectoryAction };

function emitCaptureEvent(token: MovementToken, ts: number): EmittedEvent {
  if (token.modality === "window") {
    const event = token.verb === "focus" ? "focus-changed" : "window-opened";
    return {
      kind: "observation",
      observation: {
        kind: "observation",
        source: "os",
        summary: `${token.verb} ${token.target}`,
        ts,
        metadata: { event, windowTitle: token.target },
      },
    };
  }
  if (token.modality === "command") {
    if (token.verb === "open") {
      return {
        kind: "observation",
        observation: {
          kind: "observation",
          source: "os",
          summary: `open ${token.target}`,
          ts,
          metadata: { event: "file-opened", filePath: token.target },
        },
      };
    }
    return {
      kind: "observation",
      observation: {
        kind: "observation",
        source: "os",
        summary: `run ${token.target}`,
        ts,
        metadata: { event: "command-ran", commandSummary: token.target },
      },
    };
  }

  // pointer / keyboard / gesture map to device actions.
  const gesture = deviceGestureFor(token);
  return {
    kind: "action",
    action: {
      kind: "action",
      tool: "device",
      summary: `${gesture} ${token.target}`,
      ts,
      metadata: {
        gesture,
        ...(token.target !== "*" ? { target: token.target } : {}),
        ...(token.detail ? { direction: token.detail } : {}),
      },
    },
  };
}

function deviceGestureFor(token: MovementToken): string {
  if (token.modality === "pointer") {
    return token.verb; // tap | swipe | scroll
  }
  if (token.modality === "keyboard") {
    return token.verb; // type | shortcut
  }
  return token.verb;
}
