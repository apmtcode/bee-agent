/**
 * Synthetic movement-stream generator + generalization eval harness.
 *
 * The subsystem must be validated without a real OS: this produces deterministic
 * families of related movement sequences (a "task" grammar with per-instance
 * variation) so we can (1) train a policy on some instances and (2) measure how
 * well it repeats held-out instances and generalizes to unseen-but-related ones.
 *
 * Determinism is guaranteed by an explicit seed + a small pure PRNG, so eval
 * numbers are reproducible in CI and never depend on `Math.random`.
 */

import type {
  MovementDataset,
  MovementPolicyBackend,
  MovementPolicyModel,
  MovementSequence,
  MovementToken,
} from "./movement-policy.js";

/** Mulberry32 — a tiny deterministic PRNG. Pure, seedable, no globals. */
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

/** One task template: a canonical movement plus interchangeable variants. */
export type MovementTaskTemplate = {
  intent: string;
  /** Ordered steps; each step is a set of interchangeable-but-related tokens. */
  steps: MovementToken[][];
};

export type SyntheticDatasetOptions = {
  seed: number;
  /** How many instances to emit per template. */
  instancesPerTemplate?: number;
  templates?: MovementTaskTemplate[];
};

/**
 * The default task family models a few realistic desktop workflows. Each step
 * offers a couple of related tokens (e.g. two ways to focus a field), so
 * instances share structure but differ in surface form — exactly the setting
 * where backoff generalization should help.
 */
export const DEFAULT_MOVEMENT_TEMPLATES: MovementTaskTemplate[] = [
  {
    intent: "send-message",
    steps: [
      ["app:focus:composer", "app:focus:message-box"],
      ["keyboard:type:greeting", "keyboard:type:body"],
      ["mouse:click:send", "keyboard:shortcut:send"],
    ],
  },
  {
    intent: "open-file",
    steps: [
      ["mouse:click:menu-file", "keyboard:shortcut:open"],
      ["app:focus:file-search", "keyboard:type:filename"],
      ["mouse:click:open-button", "keyboard:shortcut:confirm"],
    ],
  },
  {
    intent: "save-and-close",
    steps: [
      ["keyboard:shortcut:save", "mouse:click:save"],
      ["mouse:click:close-tab", "keyboard:shortcut:close"],
    ],
  },
];

/** Deterministically build a dataset by sampling variants from each template. */
export function generateSyntheticMovementDataset(options: SyntheticDatasetOptions): MovementDataset {
  const templates = options.templates ?? DEFAULT_MOVEMENT_TEMPLATES;
  const instances = Math.max(1, options.instancesPerTemplate ?? 6);
  const rng = mulberry32(options.seed);
  const sequences: MovementSequence[] = [];

  for (const template of templates) {
    for (let i = 0; i < instances; i += 1) {
      // A per-instance "style" scalar correlates the step choices, so a shared
      // prefix genuinely predicts later moves (real learnable structure) — with
      // occasional per-step noise so the policy is not trivially perfect.
      const style = rng();
      const tokens = template.steps.map((choices) => {
        const scalar = rng() < 0.15 ? rng() : style;
        const index = Math.min(choices.length - 1, Math.floor(scalar * choices.length));
        return choices[index] ?? choices[0]!;
      });
      sequences.push({
        id: `${template.intent}-${options.seed}-${i}`,
        tokens,
        intent: template.intent,
      });
    }
  }

  return { version: 1, sequences };
}

/** Split a dataset into train / held-out partitions deterministically. */
export function splitMovementDataset(
  dataset: MovementDataset,
  heldOutRatio = 0.3,
): { train: MovementDataset; heldOut: MovementDataset } {
  const ratio = Math.min(1, Math.max(0, heldOutRatio));
  const train: MovementSequence[] = [];
  const heldOut: MovementSequence[] = [];
  // Deterministic round-robin by index keeps both partitions structurally
  // representative without needing a shuffle.
  const stride = ratio === 0 ? Infinity : Math.max(1, Math.round(1 / ratio));
  dataset.sequences.forEach((sequence, index) => {
    if (ratio > 0 && index % stride === 0) {
      heldOut.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return {
    train: { version: 1, sequences: train },
    heldOut: { version: 1, sequences: heldOut },
  };
}

export type MovementPolicyEvaluation = {
  sequenceCount: number;
  tokenCount: number;
  /** Fraction of positions where the argmax prediction matched the actual move. */
  nextTokenAccuracy: number;
  /** Fraction of held-out sequences the policy reproduces exactly from its start. */
  exactReplayRate: number;
  /**
   * Fraction of predictions that required backing off below full order — a proxy
   * for how much *generalization* (vs. memorized replay) the eval exercised.
   */
  backoffRate: number;
};

/**
 * Measure replay + generalization fidelity of a trained policy on held-out
 * sequences. `nextTokenAccuracy` scores step-by-step prediction; `exactReplayRate`
 * scores full autoregressive rollout from each sequence's first move.
 */
export function evaluateMovementPolicy(
  backend: MovementPolicyBackend,
  model: MovementPolicyModel,
  heldOut: MovementDataset,
): MovementPolicyEvaluation {
  let correct = 0;
  let predictions = 0;
  let backoffs = 0;
  let exactReplays = 0;

  for (const sequence of heldOut.sequences) {
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const [best] = backend.predictNext(model, context);
      predictions += 1;
      if (best) {
        if (best.token === sequence.tokens[i]) {
          correct += 1;
        }
        if (best.backoffOrder < Math.min(model.order, context.length)) {
          backoffs += 1;
        }
      }
    }

    if (sequence.tokens.length > 0) {
      const rolled = backend.generate(model, { seed: sequence.tokens.slice(0, 1) });
      if (tokensEqual(rolled, sequence.tokens)) {
        exactReplays += 1;
      }
    }
  }

  return {
    sequenceCount: heldOut.sequences.length,
    tokenCount: predictions,
    nextTokenAccuracy: predictions === 0 ? 0 : correct / predictions,
    exactReplayRate: heldOut.sequences.length === 0 ? 0 : exactReplays / heldOut.sequences.length,
    backoffRate: predictions === 0 ? 0 : backoffs / predictions,
  };
}

function tokensEqual(a: MovementToken[], b: MovementToken[]): boolean {
  return a.length === b.length && a.every((token, index) => token === b[index]);
}
