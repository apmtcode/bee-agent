/**
 * Movement-model backend seam.
 *
 * The training runner ({@link ./runner.ts}) prepares a *plan* and launch script
 * that fine-tune a real small model on-device (MLX / axolotl). That path only
 * runs on the user's Apple-silicon machine and cannot execute in the cloud.
 *
 * This module supplies the missing *inference* half of the movement-learning
 * subsystem (standing objective #2, pieces c + d): a pluggable backend that
 * learns from reviewed movement trajectories and can then
 *   (c) **repeat** a recorded movement sequence when given its original context, and
 *   (d) **generalize** to a new-but-related context by adapting the nearest
 *       recorded sequence.
 *
 * {@link MovementModelBackend} is the seam. {@link NearestContextMovementBackend}
 * is a deterministic, dependency-free reference backend that runs anywhere — so
 * the whole capture → dataset → train → infer loop is exercised by tests in CI
 * without any OS input or Python toolchain. A real on-device backend (loading a
 * trained `model.gguf`) implements the same interface and is swapped in locally.
 */

/** A single observation or action within a recorded movement sequence. */
export type MovementStep =
  | { kind: "observation"; source: string; summary: string }
  | { kind: "action"; tool: string; summary: string };

/** One training example: the ordered movement sequence for a goal/context. */
export type MovementExample = {
  trajectoryId: string;
  /** Natural-language description of the goal / starting state. */
  context: string;
  steps: MovementStep[];
};

export type MovementDataset = {
  examples: MovementExample[];
};

export type MovementPrediction = {
  /** The predicted movement sequence to perform. */
  steps: MovementStep[];
  /** Id of the recorded trajectory this prediction was derived from, if any. */
  sourceTrajectoryId: string | null;
  /** Context-match similarity in [0, 1]. */
  confidence: number;
  /**
   * `false` when the context matched a recording closely enough to replay it
   * verbatim; `true` when the nearest recording was *adapted* to a new context.
   */
  generalized: boolean;
  /** Slot substitutions applied when generalizing (`from` → `to`). */
  substitutions: Array<{ from: string; to: string }>;
};

export interface MovementModel {
  /** Name of the backend that produced this model. */
  readonly backend: string;
  /** Number of examples the model was trained on. */
  readonly exampleCount: number;
  /** Predict the movement sequence for a given goal/context. */
  predict(context: string): MovementPrediction;
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset): MovementModel | Promise<MovementModel>;
}

/** Replay-event shape shared by {@link ReplayManifest} and the export manifest. */
type ReplayEventLike =
  | { kind: "transcript"; ts: number; role: string; content: string }
  | { kind: "observation"; ts: number; trajectoryId: string; source: string; summary: string }
  | { kind: "action"; ts: number; trajectoryId: string; tool: string; summary: string };

type ReplayLike = { events: ReplayEventLike[] };

/**
 * Derive a {@link MovementDataset} from reviewed replay manifests. Each distinct
 * `trajectoryId` in a replay becomes one example: its context is the replay's
 * user-authored transcript (the goal), and its steps are the observation/action
 * events for that trajectory in timeline order.
 */
export function buildMovementDataset(replays: ReplayLike[]): MovementDataset {
  const examples: MovementExample[] = [];

  for (const replay of replays) {
    const sorted = [...replay.events].sort((a, b) => a.ts - b.ts);
    const context = sorted
      .filter((event): event is Extract<ReplayEventLike, { kind: "transcript" }> => event.kind === "transcript")
      .filter((event) => event.role === "user")
      .map((event) => event.content.trim())
      .filter((content) => content.length > 0)
      .join(" ");

    const byTrajectory = new Map<string, MovementStep[]>();
    for (const event of sorted) {
      if (event.kind === "observation") {
        appendStep(byTrajectory, event.trajectoryId, { kind: "observation", source: event.source, summary: event.summary });
      } else if (event.kind === "action") {
        appendStep(byTrajectory, event.trajectoryId, { kind: "action", tool: event.tool, summary: event.summary });
      }
    }

    for (const [trajectoryId, steps] of byTrajectory) {
      if (steps.length > 0) {
        examples.push({ trajectoryId, context, steps });
      }
    }
  }

  return { examples };
}

function appendStep(map: Map<string, MovementStep[]>, key: string, step: MovementStep): void {
  const existing = map.get(key);
  if (existing) {
    existing.push(step);
  } else {
    map.set(key, [step]);
  }
}

export type NearestContextBackendOptions = {
  /**
   * Similarity at/above which the nearest recording is replayed verbatim rather
   * than generalized. Default `0.999` (i.e. an effectively identical context).
   */
  exactMatchThreshold?: number;
};

/**
 * Deterministic reference backend. It indexes examples by the token set of their
 * context, and answers a query by locating the most similar recorded context
 * (Jaccard similarity). An exact/near-exact match replays the recording; a
 * partial match is *generalized* by substituting the context tokens that differ
 * (the varying "slots", e.g. a filename or a target button) into the recorded
 * step summaries. No randomness, no I/O — same dataset + query ⇒ same output.
 */
export class NearestContextMovementBackend implements MovementModelBackend {
  readonly name = "nearest-context";

  constructor(private readonly options: NearestContextBackendOptions = {}) {}

  train(dataset: MovementDataset): MovementModel {
    const exactMatchThreshold = this.options.exactMatchThreshold ?? 0.999;
    const indexed = dataset.examples.map((example) => ({
      example,
      tokens: tokenize(example.context),
    }));
    const backend = this.name;

    return {
      backend,
      exampleCount: indexed.length,
      predict(context: string): MovementPrediction {
        const queryTokens = tokenize(context);
        let best: (typeof indexed)[number] | undefined;
        let bestScore = -1;

        for (const candidate of indexed) {
          const score = jaccard(queryTokens.set, candidate.tokens.set);
          // Deterministic tie-break: higher score wins; on a tie prefer the
          // lexicographically smaller trajectoryId.
          if (
            score > bestScore ||
            (score === bestScore && best && candidate.example.trajectoryId < best.example.trajectoryId)
          ) {
            best = candidate;
            bestScore = score;
          }
        }

        if (!best || bestScore <= 0) {
          return { steps: [], sourceTrajectoryId: null, confidence: Math.max(bestScore, 0), generalized: false, substitutions: [] };
        }

        if (bestScore >= exactMatchThreshold) {
          return {
            steps: best.example.steps.map(cloneStep),
            sourceTrajectoryId: best.example.trajectoryId,
            confidence: bestScore,
            generalized: false,
            substitutions: [],
          };
        }

        const substitutions = deriveSubstitutions(best.tokens, queryTokens);
        return {
          steps: best.example.steps.map((step) => applySubstitutions(step, substitutions)),
          sourceTrajectoryId: best.example.trajectoryId,
          confidence: bestScore,
          generalized: true,
          substitutions,
        };
      },
    };
  }
}

type Tokenized = {
  /** Lowercased token set, for similarity + slot detection. */
  set: Set<string>;
  /** Ordered lowercased tokens with the first original-cased spelling seen. */
  ordered: Array<{ lower: string; original: string }>;
};

function tokenize(text: string): Tokenized {
  const set = new Set<string>();
  const ordered: Array<{ lower: string; original: string }> = [];
  for (const raw of text.split(/[^A-Za-z0-9.]+/)) {
    const original = raw.trim();
    if (!original) {
      continue;
    }
    const lower = original.toLowerCase();
    if (!set.has(lower)) {
      set.add(lower);
      ordered.push({ lower, original });
    }
  }
  return { set, ordered };
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const token of a) {
    if (b.has(token)) {
      intersection += 1;
    }
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

/**
 * Pair up the tokens that differ between the recorded context and the query.
 * Tokens present only in the recording are the values to replace; tokens present
 * only in the query are the new values. They are paired positionally (both in
 * first-seen order) so the substitution is deterministic.
 */
function deriveSubstitutions(example: Tokenized, query: Tokenized): Array<{ from: string; to: string }> {
  const exampleOnly = example.ordered.filter((token) => !query.set.has(token.lower));
  const queryOnly = query.ordered.filter((token) => !example.set.has(token.lower));
  const pairCount = Math.min(exampleOnly.length, queryOnly.length);
  const substitutions: Array<{ from: string; to: string }> = [];
  for (let index = 0; index < pairCount; index += 1) {
    substitutions.push({ from: exampleOnly[index].original, to: queryOnly[index].original });
  }
  return substitutions;
}

function applySubstitutions(step: MovementStep, substitutions: Array<{ from: string; to: string }>): MovementStep {
  const summary = substitutions.reduce((text, { from, to }) => replaceWholeWord(text, from, to), step.summary);
  return step.kind === "observation"
    ? { kind: "observation", source: step.source, summary }
    : { kind: "action", tool: step.tool, summary };
}

function replaceWholeWord(text: string, from: string, to: string): string {
  if (!from) {
    return text;
  }
  const pattern = new RegExp(`\\b${escapeRegExp(from)}\\b`, "gi");
  return text.replace(pattern, to);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function cloneStep(step: MovementStep): MovementStep {
  return step.kind === "observation"
    ? { kind: "observation", source: step.source, summary: step.summary }
    : { kind: "action", tool: step.tool, summary: step.summary };
}
