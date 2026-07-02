import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

// ---------------------------------------------------------------------------
// Movement policy model
//
// This module closes objectives 2(c) "repeat recorded movements" and 2(d)
// "generalize to new but related movements" of the local-movement learning
// subsystem. Unlike `src/training/runner.ts` — which only *emits launch plans*
// for external on-device trainers (mlx/axolotl) that cannot execute in the
// cloud — everything here is pure, deterministic in-memory computation, so the
// capture -> dataset -> train -> predict -> eval loop runs (and is tested) in
// CI with synthetic event streams.
//
// The model is a backoff n-gram over *shape tokens* (channel + verb, with the
// concrete object stripped out) paired with slot-filling at inference time:
//   - Exact repeat fidelity: a context seen in training re-derives the recorded
//     movement verbatim (shape match + slot fill from the same object).
//   - Generalization: because context keys are object-free, the learned shape
//     transfers to an unseen-but-structurally-identical trajectory, and the
//     object slot is filled from that trajectory's own recent context.
//
// The backend is pluggable (`MovementPolicyBackend`); `NgramMovementBackend` is
// the deterministic reference/mock backend. A real on-device small model can
// implement the same interface behind this seam.
// ---------------------------------------------------------------------------

const TOKEN_FIELD_SEP = "␟"; // ␟ — separates fields within one token
const TOKEN_JOIN_SEP = "␞"; // ␞ — joins tokens into an n-gram key

/** A structural, object-free descriptor of one timeline event. */
export type MovementShapeToken = string;

/** Ordered context of preceding shape tokens (oldest -> newest). */
export type MovementContext = {
  tokens: MovementShapeToken[];
  /** The most recent salient object seen in the context, if any. */
  object?: string;
};

/** The concrete movement to predict/execute. */
export type MovementLabel = {
  tool: string;
  /** Object-free verb, e.g. "save". */
  verb: string;
  /** Concrete object slot, e.g. "report". Absent for object-less movements. */
  object?: string;
};

export type MovementSample = {
  context: MovementContext;
  label: MovementLabel;
};

export type MovementPrediction = {
  tool: string;
  verb: string;
  /** Concrete rendered movement summary after slot filling. */
  summary: string;
  object?: string;
  /** 0..1 — winning label count / total for the matched context key. */
  confidence: number;
  /** How many context tokens matched. order = exact; 0 = global prior. */
  backoffOrder: number;
  /** Observed count supporting the winning label. */
  support: number;
};

export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  entries: Array<{
    key: string;
    labels: Array<{ tool: string; verb: string; count: number }>;
  }>;
};

/**
 * Pluggable local movement policy backend. `NgramMovementBackend` is the
 * deterministic reference implementation; a real on-device model can satisfy
 * the same contract behind this seam.
 */
export interface MovementPolicyBackend {
  readonly backendId: string;
  train(samples: MovementSample[]): void;
  predict(context: MovementContext): MovementPrediction | undefined;
  snapshot(): MovementModelSnapshot;
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

/**
 * Split a summary into a verb (first word) and an object (the remainder).
 * Overridable for real capture data via `TokenizeOptions.extractVerbObject`.
 */
export function defaultVerbObject(summary: string): { verb: string; object?: string } {
  const trimmed = summary.trim();
  if (trimmed.length === 0) {
    return { verb: "" };
  }
  const spaceIndex = trimmed.indexOf(" ");
  if (spaceIndex === -1) {
    return { verb: trimmed.toLowerCase() };
  }
  const verb = trimmed.slice(0, spaceIndex).toLowerCase();
  const object = trimmed.slice(spaceIndex + 1).trim();
  return object.length > 0 ? { verb, object } : { verb };
}

export type TokenizeOptions = {
  extractVerbObject?: (summary: string) => { verb: string; object?: string };
};

type TokenizedEvent = {
  shape: MovementShapeToken;
  object?: string;
  /** Present only for action events — the label to learn/predict. */
  label?: MovementLabel;
};

function tokenizeEvent(event: ReplayTimelineEvent, options: TokenizeOptions): TokenizedEvent {
  const extract = options.extractVerbObject ?? defaultVerbObject;
  switch (event.kind) {
    case "transcript":
      // Content is intentionally dropped: only the role shapes context, keeping
      // keys stable and free of noisy free-text.
      return { shape: ["msg", event.role, ""].join(TOKEN_FIELD_SEP) };
    case "observation": {
      const { verb, object } = extract(event.summary);
      return {
        shape: ["obs", event.source, verb].join(TOKEN_FIELD_SEP),
        ...(object ? { object } : {}),
      };
    }
    case "action": {
      const { verb, object } = extract(event.summary);
      return {
        shape: ["act", event.tool, verb].join(TOKEN_FIELD_SEP),
        ...(object ? { object } : {}),
        label: { tool: event.tool, verb, ...(object ? { object } : {}) },
      };
    }
  }
}

// ---------------------------------------------------------------------------
// Dataset construction
// ---------------------------------------------------------------------------

export type BuildMovementSamplesOptions = TokenizeOptions & {
  /** Max preceding tokens retained as context (default 6). */
  contextWindow?: number;
};

/**
 * Walk a replay timeline and emit one training sample per action event: the
 * context is the window of preceding shape tokens (with the most recent object
 * recorded for slot filling), the label is the action's tool/verb/object.
 */
export function buildMovementSamples(
  manifest: ReplayManifest,
  options: BuildMovementSamplesOptions = {},
): MovementSample[] {
  const contextWindow = options.contextWindow ?? 6;
  const samples: MovementSample[] = [];
  const history: MovementShapeToken[] = [];
  let recentObject: string | undefined;

  for (const event of manifest.events) {
    const tokenized = tokenizeEvent(event, options);
    if (tokenized.label) {
      const window = history.slice(-contextWindow);
      samples.push({
        context: { tokens: window, ...(recentObject ? { object: recentObject } : {}) },
        label: tokenized.label,
      });
    }
    history.push(tokenized.shape);
    if (tokenized.object) {
      recentObject = tokenized.object;
    }
  }

  return samples;
}

export function buildMovementSamplesFromReplays(
  manifests: ReplayManifest[],
  options: BuildMovementSamplesOptions = {},
): MovementSample[] {
  return manifests.flatMap((manifest) => buildMovementSamples(manifest, options));
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/** Fill the object slot from context to produce a concrete movement summary. */
export function renderMovementSummary(verb: string, object: string | undefined): string {
  return object ? `${verb} ${object}` : verb;
}

// ---------------------------------------------------------------------------
// N-gram backend (deterministic reference / mock)
// ---------------------------------------------------------------------------

function labelKey(tool: string, verb: string): string {
  return `${tool}${TOKEN_FIELD_SEP}${verb}`;
}

function contextKey(tokens: MovementShapeToken[]): string {
  return tokens.join(TOKEN_JOIN_SEP);
}

export class NgramMovementBackend implements MovementPolicyBackend {
  readonly backendId = "ngram-movement";
  private readonly order: number;
  // contextKey -> labelKey -> { tool, verb, count }
  private readonly counts = new Map<string, Map<string, { tool: string; verb: string; count: number }>>();

  constructor(order = 4) {
    this.order = Math.max(0, Math.floor(order));
  }

  train(samples: MovementSample[]): void {
    for (const sample of samples) {
      const tokens = sample.context.tokens;
      const maxOrder = Math.min(this.order, tokens.length);
      for (let k = 0; k <= maxOrder; k++) {
        const key = contextKey(k === 0 ? [] : tokens.slice(tokens.length - k));
        let bucket = this.counts.get(key);
        if (!bucket) {
          bucket = new Map();
          this.counts.set(key, bucket);
        }
        const lk = labelKey(sample.label.tool, sample.label.verb);
        const existing = bucket.get(lk);
        if (existing) {
          existing.count += 1;
        } else {
          bucket.set(lk, { tool: sample.label.tool, verb: sample.label.verb, count: 1 });
        }
      }
    }
  }

  predict(context: MovementContext): MovementPrediction | undefined {
    const tokens = context.tokens;
    const maxOrder = Math.min(this.order, tokens.length);
    for (let k = maxOrder; k >= 0; k--) {
      const key = contextKey(k === 0 ? [] : tokens.slice(tokens.length - k));
      const bucket = this.counts.get(key);
      if (!bucket || bucket.size === 0) {
        continue;
      }
      let total = 0;
      let best: { tool: string; verb: string; count: number } | undefined;
      let bestLk = "";
      for (const [lk, entry] of bucket) {
        total += entry.count;
        // Deterministic tie-break: higher count wins; ties resolved by label key.
        if (!best || entry.count > best.count || (entry.count === best.count && lk < bestLk)) {
          best = entry;
          bestLk = lk;
        }
      }
      if (!best) {
        continue;
      }
      return {
        tool: best.tool,
        verb: best.verb,
        summary: renderMovementSummary(best.verb, context.object),
        ...(context.object ? { object: context.object } : {}),
        confidence: best.count / total,
        backoffOrder: k,
        support: best.count,
      };
    }
    return undefined;
  }

  snapshot(): MovementModelSnapshot {
    const entries = [...this.counts.entries()]
      .map(([key, bucket]) => ({
        key,
        labels: [...bucket.values()]
          .map((entry) => ({ tool: entry.tool, verb: entry.verb, count: entry.count }))
          .sort((a, b) => (b.count - a.count) || labelKey(a.tool, a.verb).localeCompare(labelKey(b.tool, b.verb))),
      }))
      .sort((a, b) => a.key.localeCompare(b.key));
    return { version: 1, backendId: this.backendId, order: this.order, entries };
  }

  static fromSnapshot(snapshot: MovementModelSnapshot): NgramMovementBackend {
    const backend = new NgramMovementBackend(snapshot.order);
    for (const entry of snapshot.entries) {
      const bucket = new Map<string, { tool: string; verb: string; count: number }>();
      for (const label of entry.labels) {
        bucket.set(labelKey(label.tool, label.verb), { tool: label.tool, verb: label.verb, count: label.count });
      }
      backend.counts.set(entry.key, bucket);
    }
    return backend;
  }
}

// ---------------------------------------------------------------------------
// Evaluation harness
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  total: number;
  /** Predictions whose tool+verb+object matched the recorded label. */
  correct: number;
  accuracy: number;
  /** Samples for which the backend produced no prediction at all. */
  unpredicted: number;
  meanConfidence: number;
};

function movementMatches(prediction: MovementPrediction, label: MovementLabel): boolean {
  return (
    prediction.tool === label.tool &&
    prediction.verb === label.verb &&
    (prediction.object ?? undefined) === (label.object ?? undefined)
  );
}

/** Top-1 accuracy of a backend over held-out samples. */
export function evaluateMovementBackend(
  backend: MovementPolicyBackend,
  samples: MovementSample[],
): MovementEvalResult {
  let correct = 0;
  let unpredicted = 0;
  let confidenceSum = 0;
  for (const sample of samples) {
    const prediction = backend.predict(sample.context);
    if (!prediction) {
      unpredicted += 1;
      continue;
    }
    confidenceSum += prediction.confidence;
    if (movementMatches(prediction, sample.label)) {
      correct += 1;
    }
  }
  const total = samples.length;
  const predicted = total - unpredicted;
  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    unpredicted,
    meanConfidence: predicted === 0 ? 0 : confidenceSum / predicted,
  };
}

export type GeneralizationEvalResult = MovementEvalResult & {
  /** Held-out samples whose object never appeared in training. */
  novel: number;
  /** Of those, how many were predicted correctly (true generalization). */
  novelCorrect: number;
  /** novelCorrect / novel — generalization accuracy on unseen objects. */
  generalizationAccuracy: number;
};

/**
 * Measure generalization: accuracy restricted to held-out samples whose object
 * slot was never observed during training. A high `generalizationAccuracy`
 * means the learned movement *shapes* transferred to new-but-related targets.
 */
export function measureGeneralization(
  backend: MovementPolicyBackend,
  trainSamples: MovementSample[],
  testSamples: MovementSample[],
): GeneralizationEvalResult {
  const trainObjects = new Set<string>();
  for (const sample of trainSamples) {
    if (sample.context.object) {
      trainObjects.add(sample.context.object);
    }
    if (sample.label.object) {
      trainObjects.add(sample.label.object);
    }
  }

  const base = evaluateMovementBackend(backend, testSamples);
  let novel = 0;
  let novelCorrect = 0;
  for (const sample of testSamples) {
    const objects = [sample.context.object, sample.label.object].filter(
      (value): value is string => typeof value === "string",
    );
    const isNovel = objects.length > 0 && objects.every((value) => !trainObjects.has(value));
    if (!isNovel) {
      continue;
    }
    novel += 1;
    const prediction = backend.predict(sample.context);
    if (prediction && movementMatches(prediction, sample.label)) {
      novelCorrect += 1;
    }
  }

  return {
    ...base,
    novel,
    novelCorrect,
    generalizationAccuracy: novel === 0 ? 0 : novelCorrect / novel,
  };
}
