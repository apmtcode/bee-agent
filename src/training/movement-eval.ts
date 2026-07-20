import type {
  MovementDataset,
  MovementPolicyModel,
  MovementSequence,
  MovementToken,
} from "./movement-policy.js";

/**
 * Synthetic movement-stream generation and generalization evaluation.
 *
 * bee-agent runs in Anthropic's cloud with no access to the user's real
 * machine, so we cannot capture genuine mouse/keyboard streams here. Instead we
 * generate grammar-driven synthetic workflows to validate the whole
 * capture → dataset → train → infer loop deterministically, and measure whether
 * a trained policy generalizes to *held-out but related* sequences (drawn from
 * the same grammar with a different seed) rather than merely memorizing.
 */

// --- Deterministic PRNG (mulberry32) -------------------------------------
// Avoids Math.random so datasets and tests are fully reproducible.

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A workflow grammar: an ordered list of stages, each offering one or more
 * alternative movement tokens. Generating a sequence picks one alternative per
 * stage (optional stages may be skipped), producing structured-but-varied
 * streams that a Markov policy can learn and generalize over.
 */
export type MovementWorkflow = {
  name: string;
  stages: Array<{
    /** Candidate tokens for this stage; one is chosen per generated sequence. */
    options: MovementToken[];
    /** Probability the stage is included at all (default 1). */
    includeProbability?: number;
    /** Max times the stage may repeat (default 1). */
    maxRepeat?: number;
  }>;
};

export type SyntheticDatasetOptions = {
  seed?: number;
  sequenceCount?: number;
  workflows?: MovementWorkflow[];
};

/**
 * A small default set of realistic desktop workflows: edit-and-save,
 * search-and-open, and a browser fill-form flow. Deterministic and dependency
 * free — the same tokens repeat across sequences so structure is learnable.
 */
export function defaultMovementWorkflows(): MovementWorkflow[] {
  return [
    {
      name: "edit-and-save",
      stages: [
        { options: ["os:focus"] },
        { options: ["editor:type", "editor:paste"] },
        { options: ["editor:type"], includeProbability: 0.5, maxRepeat: 3 },
        { options: ["editor:save"] },
        { options: ["os:command"], includeProbability: 0.4 },
      ],
    },
    {
      name: "search-and-open",
      stages: [
        { options: ["os:focus"] },
        { options: ["finder:shortcut"] },
        { options: ["finder:type"] },
        { options: ["finder:tap", "finder:scroll"], maxRepeat: 2 },
        { options: ["editor:focus"] },
      ],
    },
    {
      name: "browser-fill-form",
      stages: [
        { options: ["browser:focus"] },
        { options: ["browser:tap"] },
        { options: ["browser:type"], maxRepeat: 3 },
        { options: ["browser:tap", "browser:shortcut"] },
        { options: ["browser:type"], includeProbability: 0.6 },
        { options: ["browser:save"] },
      ],
    },
  ];
}

/** Generate a reproducible dataset of synthetic movement sequences. */
export function generateSyntheticMovementDataset(options: SyntheticDatasetOptions = {}): MovementDataset {
  const seed = options.seed ?? 1;
  const count = options.sequenceCount ?? 60;
  const workflows = options.workflows ?? defaultMovementWorkflows();
  const rand = mulberry32(seed);
  const sequences: MovementSequence[] = [];

  for (let i = 0; i < count; i += 1) {
    const workflow = workflows[Math.floor(rand() * workflows.length)] as MovementWorkflow;
    const tokens: MovementToken[] = [];
    for (const stage of workflow.stages) {
      if (stage.includeProbability !== undefined && rand() >= stage.includeProbability) {
        continue;
      }
      const repeats = 1 + Math.floor(rand() * Math.max(1, stage.maxRepeat ?? 1));
      for (let r = 0; r < repeats; r += 1) {
        const option = stage.options[Math.floor(rand() * stage.options.length)] as MovementToken;
        tokens.push(option);
      }
    }
    sequences.push({ id: `${workflow.name}-${seed}-${i}`, tokens, metadata: { workflow: workflow.name } });
  }

  return { sequences };
}

export type NextActionEvalResult = {
  /** Number of scored prediction steps (one per token plus the end marker). */
  steps: number;
  correct: number;
  /** Top-1 next-action accuracy in [0, 1]. */
  accuracy: number;
  /** Fraction of steps where the model backed off below its max order. */
  backoffRate: number;
};

/**
 * Next-action accuracy: at every position the model is asked to predict the
 * actual next token from the true prefix (teacher forcing), including the final
 * end-of-sequence prediction. This is the core generalization metric — run it
 * on held-out sequences the model never trained on.
 */
export function evaluateNextActionAccuracy(
  model: MovementPolicyModel,
  heldOut: MovementSequence[],
  maxOrderHint?: number,
): NextActionEvalResult {
  let steps = 0;
  let correct = 0;
  let backoff = 0;
  for (const sequence of heldOut) {
    for (let i = 0; i <= sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const expected: MovementToken | null = i < sequence.tokens.length ? (sequence.tokens[i] as MovementToken) : null;
      const prediction = model.predict(context);
      steps += 1;
      if (prediction.token === expected) {
        correct += 1;
      }
      if (maxOrderHint !== undefined && prediction.order >= 0 && prediction.order < Math.min(maxOrderHint, i)) {
        backoff += 1;
      }
    }
  }
  return {
    steps,
    correct,
    accuracy: steps === 0 ? 0 : correct / steps,
    backoffRate: steps === 0 ? 0 : backoff / steps,
  };
}

export type ReplayFidelityResult = {
  sequences: number;
  /** Mean over sequences of the longest-common-prefix ratio (generated vs true). */
  meanPrefixFidelity: number;
  /** Fraction of held-out sequences reproduced exactly from their first token. */
  exactMatchRate: number;
};

/**
 * Replay fidelity: seed the model with each held-out sequence's first token,
 * let it generate autonomously (argmax), and compare the generated trajectory
 * to the true one. Measures how faithfully the policy *repeats* recorded
 * movements when driving itself, not just single-step accuracy.
 */
export function evaluateReplayFidelity(
  model: MovementPolicyModel,
  heldOut: MovementSequence[],
): ReplayFidelityResult {
  if (heldOut.length === 0) {
    return { sequences: 0, meanPrefixFidelity: 0, exactMatchRate: 0 };
  }
  let prefixSum = 0;
  let exact = 0;
  for (const sequence of heldOut) {
    if (sequence.tokens.length === 0) {
      prefixSum += 1;
      exact += 1;
      continue;
    }
    const seed = sequence.tokens.slice(0, 1);
    const generated = model.generate(seed, { maxLength: sequence.tokens.length + 8 });
    const common = longestCommonPrefix(generated, sequence.tokens);
    prefixSum += common / sequence.tokens.length;
    if (generated.length === sequence.tokens.length && common === sequence.tokens.length) {
      exact += 1;
    }
  }
  return {
    sequences: heldOut.length,
    meanPrefixFidelity: prefixSum / heldOut.length,
    exactMatchRate: exact / heldOut.length,
  };
}

function longestCommonPrefix(a: MovementToken[], b: MovementToken[]): number {
  const len = Math.min(a.length, b.length);
  let i = 0;
  while (i < len && a[i] === b[i]) {
    i += 1;
  }
  return i;
}
