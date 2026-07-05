import {
  MOVEMENT_SEQUENCE_START,
  buildMovementDataset,
  tokenizeSequence,
  type MovementDataset,
  type MovementEvent,
  type MovementEventKind,
  type MovementSequence,
} from "./movement-event.js";
import { createSeededRng, type MovementModelBackend } from "./model-backend.js";

/**
 * Synthetic movement-stream generator + generalization eval harness.
 *
 * The engine has no real machine to capture from, so we validate the whole
 * capture-schema → dataset → train → generate loop against *simulated* movement
 * streams. Sequences are drawn from intent templates (open an app, click a
 * control, type, save, …) with seeded, bounded variation, so the model sees a
 * realistic distribution of related-but-not-identical trajectories — exactly the
 * regime where "repeat the recorded movement" and "generalize to a new but
 * related movement" must both hold.
 */

export type MovementTemplate = {
  intent: string;
  appId: string;
  /** Ordered steps; each contributes one movement event. */
  steps: MovementTemplateStep[];
};

export type MovementTemplateStep = {
  kind: MovementEventKind;
  /** Candidate targets — one is chosen per instance via the seeded RNG. */
  targets?: string[];
  key?: string;
  direction?: MovementEvent["direction"];
  /** Probability in [0,1] this optional step is included. Default 1. */
  probability?: number;
};

export type SyntheticDatasetParams = {
  templates: MovementTemplate[];
  /** Instances to generate per template. */
  instancesPerTemplate: number;
  seed: number;
  /** Milliseconds between consecutive events. */
  stepIntervalMs?: number;
};

export function generateSyntheticMovementDataset(params: SyntheticDatasetParams): MovementDataset {
  const rng = createSeededRng(params.seed);
  const interval = params.stepIntervalMs ?? 120;
  const sequences: MovementSequence[] = [];

  params.templates.forEach((template, templateIndex) => {
    for (let instance = 0; instance < params.instancesPerTemplate; instance += 1) {
      const events: MovementEvent[] = [];
      let ts = 0;
      for (const step of template.steps) {
        const probability = step.probability ?? 1;
        if (probability < 1 && rng() >= probability) {
          continue;
        }
        const event: MovementEvent = { kind: step.kind, ts, appId: template.appId };
        if (step.targets && step.targets.length > 0) {
          const index = Math.floor(rng() * step.targets.length) % step.targets.length;
          event.target = step.targets[index];
        }
        if (step.key) {
          event.key = step.key;
        }
        if (step.direction) {
          event.direction = step.direction;
        }
        events.push(event);
        ts += interval;
      }
      sequences.push({
        id: `${template.intent}-${templateIndex}-${instance}`,
        intent: template.intent,
        appId: template.appId,
        events,
      });
    }
  });

  return buildMovementDataset(sequences);
}

/** A small, realistic default template set for tests and demos. */
export function defaultMovementTemplates(): MovementTemplate[] {
  return [
    {
      intent: "save document",
      appId: "editor",
      steps: [
        { kind: "focus", targets: ["editor-window"] },
        { kind: "click", targets: ["document-body", "line-1", "line-2"] },
        { kind: "key-type", targets: ["document-body"] },
        { kind: "shortcut", key: "cmd+s" },
      ],
    },
    {
      intent: "search and open",
      appId: "browser",
      steps: [
        { kind: "focus", targets: ["browser-window"] },
        { kind: "click", targets: ["address-bar"] },
        { kind: "key-type", targets: ["address-bar"] },
        { kind: "shortcut", key: "enter" },
        { kind: "scroll", direction: "down", probability: 0.7 },
        { kind: "click", targets: ["result-1", "result-2", "result-3"] },
      ],
    },
    {
      intent: "reply to message",
      appId: "chat",
      steps: [
        { kind: "focus", targets: ["chat-window"] },
        { kind: "click", targets: ["thread-1", "thread-2"] },
        { kind: "click", targets: ["composer"] },
        { kind: "key-type", targets: ["composer"] },
        { kind: "shortcut", key: "cmd+enter" },
      ],
    },
  ];
}

export type GeneralizationReport = {
  trainSequences: number;
  heldOutSequences: number;
  /** Next-token argmax accuracy on held-out prefixes. */
  nextTokenAccuracy: number;
  /** Fraction of held-out bigrams the model assigns non-trivial mass to. */
  transitionCoverage: number;
  /** Fraction of held-out sequences reproduced exactly by greedy decoding. */
  exactSequenceMatch: number;
  evaluatedTokens: number;
};

export type EvaluateGeneralizationParams = {
  backend: MovementModelBackend;
  dataset: MovementDataset;
  /** Fraction of sequences held out for evaluation. Default 0.3. */
  holdoutRatio?: number;
  seed: number;
  order?: number;
  smoothing?: number;
};

/**
 * Split the dataset into train/held-out, train the backend on the train split,
 * then measure how well it predicts and reproduces the held-out (unseen)
 * sequences. High next-token accuracy with a non-trivial hold-out set is
 * evidence the model generalizes rather than memorizes.
 */
export async function evaluateGeneralization(
  params: EvaluateGeneralizationParams,
): Promise<GeneralizationReport> {
  const holdoutRatio = clamp(params.holdoutRatio ?? 0.3, 0, 0.9);
  const shuffled = deterministicShuffle(params.dataset.sequences, params.seed);
  const holdoutCount = Math.min(
    Math.max(1, Math.round(shuffled.length * holdoutRatio)),
    Math.max(0, shuffled.length - 1),
  );
  const heldOut = shuffled.slice(0, holdoutCount);
  const train = shuffled.slice(holdoutCount);

  const model = await params.backend.train(buildMovementDataset(train), {
    ...(params.order !== undefined ? { order: params.order } : {}),
    ...(params.smoothing !== undefined ? { smoothing: params.smoothing } : {}),
  });

  let correct = 0;
  let evaluated = 0;
  let coveredBigrams = 0;
  let totalBigrams = 0;
  let exactMatches = 0;

  const startPad = Array<string>(model.metadata.order).fill(MOVEMENT_SEQUENCE_START);
  for (const sequence of heldOut) {
    const tokens = tokenizeSequence(sequence);
    for (let i = 0; i < tokens.length; i += 1) {
      // Pad with START so position 0 is a start-of-sequence prediction, matching
      // how generation conditions — otherwise the first token has no context.
      const context = [...startPad, ...tokens.slice(0, i)];
      const distribution = model.predictNext(context);
      const predicted = distribution[0]?.token;
      if (predicted === tokens[i]) {
        correct += 1;
      }
      evaluated += 1;
      if (i > 0) {
        totalBigrams += 1;
        const mass = distribution.find((entry) => entry.token === tokens[i])?.probability ?? 0;
        if (mass > 0) {
          coveredBigrams += 1;
        }
      }
    }
    const generated = model.generate({ maxLength: tokens.length + 4 });
    if (arraysEqual(generated, tokens)) {
      exactMatches += 1;
    }
  }

  return {
    trainSequences: train.length,
    heldOutSequences: heldOut.length,
    nextTokenAccuracy: evaluated === 0 ? 0 : correct / evaluated,
    transitionCoverage: totalBigrams === 0 ? 0 : coveredBigrams / totalBigrams,
    exactSequenceMatch: heldOut.length === 0 ? 0 : exactMatches / heldOut.length,
    evaluatedTokens: evaluated,
  };
}

function deterministicShuffle<T>(items: T[], seed: number): T[] {
  const rng = createSeededRng(seed);
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    const a = copy[i]!;
    copy[i] = copy[j]!;
    copy[j] = a;
  }
  return copy;
}

function arraysEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((value, index) => value === b[index]);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
