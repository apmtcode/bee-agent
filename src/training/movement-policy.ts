import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning model.
 *
 * This is the on-device "post-trained model" half of the movement-learning
 * subsystem (standing objective #2c/#2d). It learns from recorded movement
 * trajectories — ordered streams of observations and actions — and can then:
 *
 *   (c) *repeat* recorded movements: replay the dominant learned action
 *       sequence from a seed observation, and
 *   (d) *generalize*: pick a sensible action for a new-but-related observation
 *       it never saw verbatim, via shared observation cues.
 *
 * The model is a small, fully deterministic, JSON-serializable order-k Markov
 * model over action tokens (with START/END sentinels and stupid-backoff
 * interpolation) plus an observation→action cue table. It needs no native
 * code, so it trains and runs in the cloud/CI for tests; the heavyweight
 * on-device backends (mlx/axolotl) remain a pluggable seam — see backend.ts.
 */

export type MovementObservation = {
  source: string;
  summary: string;
};

export type MovementAction = {
  tool: string;
  summary: string;
};

export type MovementEvent =
  | ({ kind: "observation" } & MovementObservation)
  | ({ kind: "action" } & MovementAction);

export type MovementTrajectory = {
  id: string;
  events: MovementEvent[];
};

export type MovementPolicy = {
  version: 1;
  order: number;
  /** Action tokens seen in training, excluding START/END sentinels. */
  vocabulary: string[];
  /** Decode an action token back into its tool/summary pair. */
  tokenActions: Record<string, MovementAction>;
  /** order-m context key -> next action token -> observed count. */
  transitions: Record<string, Record<string, number>>;
  /** observation feature -> action token -> observed count. */
  observationCues: Record<string, Record<string, number>>;
  trainedTrajectories: number;
  trainedActions: number;
};

export type MovementPrediction = {
  action: MovementAction;
  token: string;
  /** Normalized score of the chosen token within the candidate set (0..1). */
  confidence: number;
  /** True when the model predicts the end of the movement sequence. */
  end: boolean;
};

export type PredictContext = {
  recentActions?: MovementAction[];
  observation?: MovementObservation;
};

export type ReplayOptions = {
  observation?: MovementObservation;
  maxSteps?: number;
  minConfidence?: number;
};

const START = "START";
const END = "END";
const SEP = "␞";
const FIELD = "␟";

const DEFAULT_ORDER = 3;
/** Weight applied to observation-cue scores relative to Markov scores. */
const OBSERVATION_WEIGHT = 4;

function actionToken(action: MovementAction): string {
  return `${action.tool}${FIELD}${action.summary}`;
}

function normalizeWord(word: string): string {
  return word.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

/** Extract generalization features from an observation (source + content words). */
function observationFeatures(observation: MovementObservation): string[] {
  const features = new Set<string>();
  const source = observation.source.trim().toLowerCase();
  if (source) {
    features.add(`src${FIELD}${source}`);
  }
  for (const raw of observation.summary.split(/\s+/)) {
    const word = normalizeWord(raw);
    if (word.length >= 3) {
      features.add(`w${FIELD}${word}`);
    }
  }
  return [...features];
}

function contextKey(order: number, recentTokens: string[]): string {
  const slice = recentTokens.slice(-order);
  while (slice.length < order) {
    slice.unshift(START);
  }
  return `${order}|${slice.join(SEP)}`;
}

function increment(table: Record<string, Record<string, number>>, key: string, token: string): void {
  const row = (table[key] ??= {});
  row[token] = (row[token] ?? 0) + 1;
}

/** Convert capture replay-manifest events into a normalized movement trajectory. */
export function movementTrajectoryFromReplay(id: string, events: ReplayTimelineEvent[]): MovementTrajectory {
  const movementEvents: MovementEvent[] = [];
  for (const event of events) {
    if (event.kind === "observation") {
      movementEvents.push({ kind: "observation", source: event.source, summary: event.summary });
    } else if (event.kind === "action") {
      movementEvents.push({ kind: "action", tool: event.tool, summary: event.summary });
    }
  }
  return { id, events: movementEvents };
}

export type TrainMovementPolicyOptions = {
  order?: number;
};

export function trainMovementPolicy(
  trajectories: MovementTrajectory[],
  options: TrainMovementPolicyOptions = {},
): MovementPolicy {
  const order = Math.max(1, options.order ?? DEFAULT_ORDER);
  const policy: MovementPolicy = {
    version: 1,
    order,
    vocabulary: [],
    tokenActions: {},
    transitions: {},
    observationCues: {},
    trainedTrajectories: 0,
    trainedActions: 0,
  };
  const vocabulary = new Set<string>();

  for (const trajectory of trajectories) {
    const actionTokens: string[] = [];
    let pendingObservation: MovementObservation | undefined;
    let sawAction = false;

    for (const event of trajectory.events) {
      if (event.kind === "observation") {
        pendingObservation = { source: event.source, summary: event.summary };
        continue;
      }
      sawAction = true;
      const token = actionToken(event);
      vocabulary.add(token);
      policy.tokenActions[token] = { tool: event.tool, summary: event.summary };
      policy.trainedActions += 1;

      for (let m = 0; m <= order; m += 1) {
        increment(policy.transitions, contextKey(m, actionTokens), token);
      }
      if (pendingObservation) {
        for (const feature of observationFeatures(pendingObservation)) {
          increment(policy.observationCues, feature, token);
        }
      }
      actionTokens.push(token);
    }

    if (sawAction) {
      // Teach the model where movement sequences end.
      for (let m = 0; m <= order; m += 1) {
        increment(policy.transitions, contextKey(m, actionTokens), END);
      }
      policy.trainedTrajectories += 1;
    }
  }

  policy.vocabulary = [...vocabulary].sort();
  return policy;
}

function markovScores(policy: MovementPolicy, recentTokens: string[]): Map<string, number> {
  const scores = new Map<string, number>();
  for (let m = 0; m <= policy.order; m += 1) {
    const row = policy.transitions[contextKey(m, recentTokens)];
    if (!row) {
      continue;
    }
    const total = Object.values(row).reduce((sum, count) => sum + count, 0);
    if (total === 0) {
      continue;
    }
    const weight = m + 1;
    for (const [token, count] of Object.entries(row)) {
      scores.set(token, (scores.get(token) ?? 0) + (weight * count) / total);
    }
  }
  return scores;
}

function observationScores(policy: MovementPolicy, observation: MovementObservation): Map<string, number> {
  const scores = new Map<string, number>();
  for (const feature of observationFeatures(observation)) {
    const row = policy.observationCues[feature];
    if (!row) {
      continue;
    }
    const total = Object.values(row).reduce((sum, count) => sum + count, 0);
    if (total === 0) {
      continue;
    }
    for (const [token, count] of Object.entries(row)) {
      scores.set(token, (scores.get(token) ?? 0) + (OBSERVATION_WEIGHT * count) / total);
    }
  }
  return scores;
}

/**
 * Predict the next action token given recent actions and/or an observation.
 * Deterministic: ties are broken by token order so the same model + context
 * always yields the same prediction.
 */
export function predictMovement(policy: MovementPolicy, context: PredictContext): MovementPrediction | undefined {
  const recentTokens = (context.recentActions ?? []).map(actionToken);
  const scores = markovScores(policy, recentTokens);
  if (context.observation) {
    for (const [token, value] of observationScores(policy, context.observation)) {
      scores.set(token, (scores.get(token) ?? 0) + value);
    }
  }
  if (scores.size === 0) {
    return undefined;
  }

  const total = [...scores.values()].reduce((sum, value) => sum + value, 0);
  let bestToken: string | undefined;
  let bestScore = -Infinity;
  for (const [token, value] of [...scores.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))) {
    if (value > bestScore) {
      bestScore = value;
      bestToken = token;
    }
  }
  if (bestToken === undefined) {
    return undefined;
  }

  const confidence = total > 0 ? bestScore / total : 0;
  if (bestToken === END) {
    return { action: { tool: "", summary: "" }, token: END, confidence, end: true };
  }
  const action = policy.tokenActions[bestToken] ?? decodeToken(bestToken);
  return { action, token: bestToken, confidence, end: false };
}

function decodeToken(token: string): MovementAction {
  const [tool, summary] = token.split(FIELD);
  return { tool: tool ?? "", summary: summary ?? "" };
}

/**
 * Roll out a movement sequence from a seed observation, reproducing recorded
 * movements and generalizing past the seed. Stops at the learned END marker,
 * when confidence drops below `minConfidence`, or after `maxSteps`.
 */
export function replayMovement(policy: MovementPolicy, options: ReplayOptions = {}): MovementAction[] {
  const maxSteps = options.maxSteps ?? 64;
  const minConfidence = options.minConfidence ?? 0;
  const actions: MovementAction[] = [];
  let observation = options.observation;

  for (let step = 0; step < maxSteps; step += 1) {
    const prediction = predictMovement(policy, { recentActions: actions, observation });
    observation = undefined; // the seed observation only cues the first step
    if (!prediction || prediction.end || prediction.confidence < minConfidence) {
      break;
    }
    actions.push(prediction.action);
  }
  return actions;
}

export function serializeMovementPolicy(policy: MovementPolicy): string {
  return JSON.stringify(policy, null, 2);
}

export function deserializeMovementPolicy(serialized: string): MovementPolicy {
  const parsed = JSON.parse(serialized) as MovementPolicy;
  if (parsed.version !== 1) {
    throw new Error(`Unsupported movement policy version: ${String(parsed.version)}`);
  }
  return parsed;
}
