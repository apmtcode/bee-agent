/**
 * Generalization eval harness for movement models (objective #2 d).
 *
 * Measures two things on held-out sequences the model did NOT train on:
 *  - next-event prediction accuracy (does the model pick the right movement token
 *    given each true prefix?), and
 *  - rollout fidelity (how much of a held-out sequence a free-running generate()
 *    reproduces, via longest-common-subsequence over tokens).
 *
 * Both are deterministic and pure, so the harness runs identically in CI on
 * synthetic data and locally on real captures.
 */
import {
  movementEventToken,
  type MovementEvent,
  type MovementSequence,
  type TrainedMovementModel,
} from "./movement-model.js";

export type NextEventEval = {
  predictions: number;
  correct: number;
  /** Correct-token predictions / total predictions, in [0,1]. */
  accuracy: number;
};

export type RolloutEval = {
  sequences: number;
  /** Mean token-level LCS overlap between generated and true continuations, [0,1]. */
  meanFidelity: number;
  /** Fraction of held-out sequences whose full token stream is reproduced exactly. */
  exactMatchRate: number;
};

export type GeneralizationReport = {
  nextEvent: NextEventEval;
  rollout: RolloutEval;
};

/**
 * Next-event accuracy: for each held-out sequence, walk every prefix and check the
 * predicted next token against the true next token.
 */
export function evaluateNextEvent(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): NextEventEval {
  let predictions = 0;
  let correct = 0;
  for (const sequence of heldOut) {
    for (let i = 1; i < sequence.events.length; i += 1) {
      const prefix = sequence.events.slice(0, i);
      const prediction = model.predictNext(prefix);
      predictions += 1;
      if (prediction && movementEventToken(prediction.event) === movementEventToken(sequence.events[i])) {
        correct += 1;
      }
    }
  }
  return { predictions, correct, accuracy: predictions === 0 ? 0 : correct / predictions };
}

function lcsLength(a: string[], b: string[]): number {
  const rows = a.length + 1;
  const cols = b.length + 1;
  const table = new Array<number>(rows * cols).fill(0);
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      if (a[i - 1] === b[j - 1]) {
        table[i * cols + j] = table[(i - 1) * cols + (j - 1)] + 1;
      } else {
        table[i * cols + j] = Math.max(table[(i - 1) * cols + j], table[i * cols + (j - 1)]);
      }
    }
  }
  return table[(rows - 1) * cols + (cols - 1)];
}

/**
 * Rollout fidelity: seed the model with the first `seedLength` events of each
 * held-out sequence, free-run generate(), and compare the generated continuation to
 * the true continuation by token-level LCS.
 */
export function evaluateRolloutFidelity(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
  options?: { seedLength?: number },
): RolloutEval {
  const seedLength = Math.max(1, options?.seedLength ?? 1);
  let fidelitySum = 0;
  let exact = 0;
  let scored = 0;

  for (const sequence of heldOut) {
    if (sequence.events.length <= seedLength) {
      continue;
    }
    scored += 1;
    const seed = sequence.events.slice(0, seedLength);
    const trueTail = sequence.events.slice(seedLength).map(movementEventToken);
    const generated: MovementEvent[] = model.generate(seed, { maxSteps: trueTail.length + 4 });
    const generatedTokens = generated.map(movementEventToken);

    const overlap = lcsLength(generatedTokens, trueTail);
    const fidelity = trueTail.length === 0 ? 0 : overlap / trueTail.length;
    fidelitySum += fidelity;
    if (
      generatedTokens.length === trueTail.length &&
      generatedTokens.every((token, index) => token === trueTail[index])
    ) {
      exact += 1;
    }
  }

  return {
    sequences: scored,
    meanFidelity: scored === 0 ? 0 : fidelitySum / scored,
    exactMatchRate: scored === 0 ? 0 : exact / scored,
  };
}

export function evaluateGeneralization(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
  options?: { seedLength?: number },
): GeneralizationReport {
  return {
    nextEvent: evaluateNextEvent(model, heldOut),
    rollout: evaluateRolloutFidelity(model, heldOut, options),
  };
}
