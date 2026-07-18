/**
 * Movement policy model — the local, on-device "brain" the movement-learning
 * subsystem post-trains from reviewed capture trajectories (standing objective
 * #2, parts c/d).
 *
 * This is a deterministic, fully serializable next-action model: it learns a
 * mapping from an observation context to the action that followed it, so it can
 * (c) repeat recorded movements and (d) generalize to new-but-related
 * observations via token-overlap nearest-neighbour matching. It is intentionally
 * dependency-free and pure so the whole capture -> dataset -> train -> infer loop
 * runs in the cloud/CI with no OS access; a real on-device model plugs in behind
 * the `TrainingBackend` seam (see backend.ts) while sharing this data contract.
 */

/** A single timeline event a policy learns from. Mirrors replay-manifest events
 * (minus transcript), so `ReplayTimelineEvent`/`ExportedReplayManifest.events`
 * feed in directly. */
export type MovementPolicyEvent =
  | { kind: "observation"; source: string; summary: string }
  | { kind: "action"; tool: string; summary: string };

/** One recorded/derived movement stream (ordered events). */
export type MovementReplay = {
  events: MovementPolicyEvent[];
};

export type MovementObservationInput = {
  source: string;
  summary: string;
};

export type MovementActionCandidate = {
  tool: string;
  summary: string;
  /** Number of times this action followed the observation during training. */
  count: number;
};

export type MovementTransition = {
  /** Normalized observation key (see `observationKey`). */
  key: string;
  observation: MovementObservationInput;
  /** Actions that followed this observation, ranked by descending count. */
  actions: MovementActionCandidate[];
  /** Cached token set for generalization (sorted, de-duped). */
  tokens: string[];
};

export type MovementPolicyModel = {
  version: 1;
  kind: "frequency-nextaction";
  /** Observations seen during training. */
  observationCount: number;
  /** Observation->action pairs seen during training. */
  actionCount: number;
  transitions: MovementTransition[];
  /** Marginal action frequencies — the fallback prior when nothing matches. */
  actionPrior: MovementActionCandidate[];
  vocabulary: {
    observations: string[];
    actions: string[];
  };
};

export type MovementPredictionMatch = "exact" | "generalized" | "prior" | "none";

export type MovementPrediction = {
  action?: MovementActionCandidate;
  /** All candidate actions for the matched context, ranked. */
  candidates: MovementActionCandidate[];
  match: MovementPredictionMatch;
  /** 0..1 — share of the matched context's mass on the top action, scaled by
   * match quality (token overlap) for generalized matches. */
  confidence: number;
  /** For generalized matches, the training observation that was matched. */
  matchedObservation?: MovementObservationInput;
};

export type PredictActionOptions = {
  /** Minimum Jaccard token overlap to accept a generalized match (0..1). */
  minOverlap?: number;
  /** Fall back to the marginal action prior when no context matches. */
  usePrior?: boolean;
};

const DEFAULT_MIN_OVERLAP = 0.34;

/** Normalize an observation into a stable lookup key. */
export function observationKey(observation: MovementObservationInput): string {
  return `${normalizeText(observation.source)}${normalizeText(observation.summary)}`;
}

function normalizeText(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function tokenize(observation: MovementObservationInput): string[] {
  const raw = `${observation.source} ${observation.summary}`.toLowerCase();
  const seen = new Set<string>();
  for (const token of raw.split(/[^a-z0-9]+/)) {
    if (token.length > 1) {
      seen.add(token);
    }
  }
  return [...seen].sort();
}

function jaccard(a: string[], b: string[]): number {
  if (a.length === 0 && b.length === 0) {
    return 0;
  }
  const setB = new Set(b);
  let intersection = 0;
  for (const token of a) {
    if (setB.has(token)) {
      intersection += 1;
    }
  }
  const union = a.length + b.length - intersection;
  return union === 0 ? 0 : intersection / union;
}

function rankCandidates(candidates: Map<string, MovementActionCandidate>): MovementActionCandidate[] {
  return [...candidates.values()].sort((a, b) => {
    if (b.count !== a.count) {
      return b.count - a.count;
    }
    // Stable, deterministic tiebreak so serialized models are reproducible.
    if (a.tool !== b.tool) {
      return a.tool < b.tool ? -1 : 1;
    }
    return a.summary < b.summary ? -1 : a.summary > b.summary ? 1 : 0;
  });
}

/**
 * Train a movement policy from one or more replay streams. Deterministic: the
 * same input always yields byte-identical output, so the serialized model can be
 * diffed and cached.
 */
export function trainMovementPolicy(replays: MovementReplay[]): MovementPolicyModel {
  const transitionByKey = new Map<
    string,
    { observation: MovementObservationInput; tokens: string[]; actions: Map<string, MovementActionCandidate> }
  >();
  const priorActions = new Map<string, MovementActionCandidate>();
  const observationVocab = new Set<string>();
  const actionVocab = new Set<string>();
  let observationCount = 0;
  let actionCount = 0;

  for (const replay of replays) {
    let context: MovementObservationInput | undefined;
    for (const event of replay.events) {
      if (event.kind === "observation") {
        context = { source: event.source, summary: event.summary };
        observationCount += 1;
        observationVocab.add(observationKey(context));
        continue;
      }

      // action
      actionVocab.add(`${event.tool}${normalizeText(event.summary)}`);
      recordCandidate(priorActions, event.tool, event.summary);
      if (!context) {
        continue;
      }
      actionCount += 1;
      const key = observationKey(context);
      let entry = transitionByKey.get(key);
      if (!entry) {
        entry = { observation: context, tokens: tokenize(context), actions: new Map() };
        transitionByKey.set(key, entry);
      }
      recordCandidate(entry.actions, event.tool, event.summary);
    }
  }

  const transitions: MovementTransition[] = [...transitionByKey.entries()]
    .map(([key, entry]) => ({
      key,
      observation: entry.observation,
      actions: rankCandidates(entry.actions),
      tokens: entry.tokens,
    }))
    .sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));

  return {
    version: 1,
    kind: "frequency-nextaction",
    observationCount,
    actionCount,
    transitions,
    actionPrior: rankCandidates(priorActions),
    vocabulary: {
      observations: [...observationVocab].sort(),
      actions: [...actionVocab].sort(),
    },
  };
}

function recordCandidate(target: Map<string, MovementActionCandidate>, tool: string, summary: string): void {
  const id = `${tool}${summary}`;
  const existing = target.get(id);
  if (existing) {
    existing.count += 1;
    return;
  }
  target.set(id, { tool, summary, count: 1 });
}

/**
 * Predict the next action for an observation. Tries an exact context match
 * first, then generalizes to the most token-similar seen observation, then
 * (optionally) falls back to the marginal action prior.
 */
export function predictAction(
  model: MovementPolicyModel,
  observation: MovementObservationInput,
  options: PredictActionOptions = {},
): MovementPrediction {
  const minOverlap = options.minOverlap ?? DEFAULT_MIN_OVERLAP;
  const usePrior = options.usePrior ?? true;

  const key = observationKey(observation);
  const exact = model.transitions.find((transition) => transition.key === key);
  if (exact && exact.actions.length > 0) {
    return {
      action: exact.actions[0],
      candidates: exact.actions,
      match: "exact",
      confidence: shareOfTop(exact.actions),
      matchedObservation: exact.observation,
    };
  }

  const tokens = tokenize(observation);
  let best: { transition: MovementTransition; overlap: number } | undefined;
  for (const transition of model.transitions) {
    const overlap = jaccard(tokens, transition.tokens);
    if (overlap >= minOverlap && (!best || overlap > best.overlap)) {
      best = { transition, overlap };
    }
  }
  if (best && best.transition.actions.length > 0) {
    return {
      action: best.transition.actions[0],
      candidates: best.transition.actions,
      match: "generalized",
      confidence: shareOfTop(best.transition.actions) * best.overlap,
      matchedObservation: best.transition.observation,
    };
  }

  if (usePrior && model.actionPrior.length > 0) {
    return {
      action: model.actionPrior[0],
      candidates: model.actionPrior,
      match: "prior",
      confidence: shareOfTop(model.actionPrior),
    };
  }

  return { candidates: [], match: "none", confidence: 0 };
}

function shareOfTop(candidates: MovementActionCandidate[]): number {
  const total = candidates.reduce((sum, candidate) => sum + candidate.count, 0);
  if (total === 0) {
    return 0;
  }
  return candidates[0].count / total;
}

/**
 * Replay a policy over an ordered observation stream, producing the predicted
 * action sequence. This is how a trained model "repeats" recorded movements and
 * extends to new ones.
 */
export function replayPolicy(
  model: MovementPolicyModel,
  observations: MovementObservationInput[],
  options: PredictActionOptions = {},
): MovementPrediction[] {
  return observations.map((observation) => predictAction(model, observation, options));
}

export type PolicyEvaluation = {
  total: number;
  correct: number;
  /** Top-1 accuracy over held-out (observation, expected action) pairs. */
  accuracy: number;
  matchBreakdown: Record<MovementPredictionMatch, number>;
};

/**
 * Evaluate a trained policy against held-out replays: for each observation whose
 * next event is an action, check whether the model's top prediction matches the
 * recorded action. Supports the generalization eval harness (objective #2d).
 */
export function evaluatePolicy(
  model: MovementPolicyModel,
  replays: MovementReplay[],
  options: PredictActionOptions = {},
): PolicyEvaluation {
  const matchBreakdown: Record<MovementPredictionMatch, number> = {
    exact: 0,
    generalized: 0,
    prior: 0,
    none: 0,
  };
  let total = 0;
  let correct = 0;

  for (const replay of replays) {
    let context: MovementObservationInput | undefined;
    for (const event of replay.events) {
      if (event.kind === "observation") {
        context = { source: event.source, summary: event.summary };
        continue;
      }
      if (!context) {
        continue;
      }
      total += 1;
      const prediction = predictAction(model, context, options);
      matchBreakdown[prediction.match] += 1;
      if (prediction.action && prediction.action.tool === event.tool && prediction.action.summary === event.summary) {
        correct += 1;
      }
      context = undefined;
    }
  }

  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    matchBreakdown,
  };
}
