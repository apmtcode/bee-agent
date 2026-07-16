// Deterministic mock movement-model backend.
//
// A small but *real* learning algorithm: an order-k Markov policy over
// tokenized movement events, learned per-context with graceful backoff for
// generalization. It is fully deterministic (argmax with stable lexical
// tie-breaking, no randomness, no clock), so the cloud test suite can validate
// the entire train -> replay -> generalize loop on synthetic event streams.
//
// Generalization comes from three nested distributions consulted in order:
//   1. exact context (same app + goal)  -> faithful replay of learned tasks
//   2. same app, any goal               -> related-but-new movements
//   3. global (all contexts)            -> broadest backoff
// Within each bucket the model backs off from an order-k gram down to unigram.

import {
  MOVEMENT_END_TOKEN,
  MOVEMENT_GLOBAL_KEY,
  MOVEMENT_START_TOKEN,
  movementAppKey,
  movementContextKey,
  movementEventToken,
  type MovementContext,
  type MovementEvent,
  type MovementModelBackend,
  type MovementModelTrainingConfig,
  type MovementPrediction,
  type MovementPredictionSource,
  type MovementRolloutOptions,
  type MovementTrainingDataset,
  type MovementTrajectory,
  type SerializedMovementModel,
  type TrainedMovementModel,
} from "./movement-model.js";

export const MOCK_MOVEMENT_BACKEND_ID = "mock-ngram";

const DEFAULT_ORDER = 2;
const DEFAULT_MAX_STEPS = 256;
const DEFAULT_STEP_DT = 16;

type CountMap = Map<string, number>;
type GramTable = Map<string, CountMap>;
type Template = { event: Omit<MovementEvent, "ts">; dt: number };

/** A candidate distribution plus which bucket it came from. */
type Candidate = { counts: CountMap; source: MovementPredictionSource } | undefined;

class NgramMovementModel implements TrainedMovementModel {
  readonly backendId = MOCK_MOVEMENT_BACKEND_ID;

  constructor(
    readonly order: number,
    readonly version: number,
    private readonly transitions: Map<string, GramTable>,
    private readonly templates: Map<string, Template>,
  ) {}

  predictNext(context: MovementContext, prefix: MovementEvent[]): MovementPrediction {
    const gram = prefixToGram(prefix, this.order);
    const { counts, source } = this.resolve(context, gram);
    const token = argmax(counts);
    if (token === undefined || token === MOVEMENT_END_TOKEN) {
      return { token: MOVEMENT_END_TOKEN, confidence: confidenceOf(counts, MOVEMENT_END_TOKEN), source, end: true };
    }
    const template = this.templates.get(token);
    const lastTs = prefix.length > 0 ? prefix[prefix.length - 1]!.ts : 0;
    const event = template ? materialize(template, lastTs) : undefined;
    return { token, event, confidence: confidenceOf(counts, token), source, end: false };
  }

  rollout(context: MovementContext, options: MovementRolloutOptions = {}): MovementEvent[] {
    const maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
    const events: MovementEvent[] = [];
    let ts = options.startTs ?? 0;
    let seededStart = false;
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context, events);
      if (prediction.end || !prediction.event) {
        break;
      }
      const template = this.templates.get(prediction.token)!;
      // First event anchors at startTs; subsequent events advance by learned dt.
      ts = seededStart ? ts + Math.max(1, template.dt) : ts;
      seededStart = true;
      events.push({ ...prediction.event, ts });
    }
    return events;
  }

  serialize(): SerializedMovementModel {
    const transitions: SerializedMovementModel["transitions"] = {};
    for (const [bucketKey, table] of this.transitions) {
      const serializedTable: Record<string, Record<string, number>> = {};
      for (const [gram, counts] of table) {
        serializedTable[gram] = Object.fromEntries(counts);
      }
      transitions[bucketKey] = serializedTable;
    }
    const templates: SerializedMovementModel["templates"] = {};
    for (const [token, template] of this.templates) {
      templates[token] = template;
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      modelVersion: this.version,
      transitions,
      templates,
    };
  }

  /** Consult exact context, then app, then global; back off order-k -> unigram. */
  private resolve(context: MovementContext, gram: string[]): { counts: CountMap; source: MovementPredictionSource } {
    const buckets: Array<{ key: string; source: MovementPredictionSource }> = [
      { key: movementContextKey(context), source: "context" },
      { key: movementAppKey(context), source: "app" },
      { key: MOVEMENT_GLOBAL_KEY, source: "global" },
    ];
    for (const { key, source } of buckets) {
      const candidate = this.lookup(key, gram, source);
      if (candidate) {
        return candidate;
      }
    }
    return { counts: new Map(), source: "global" };
  }

  private lookup(bucketKey: string, gram: string[], source: MovementPredictionSource): Candidate {
    const table = this.transitions.get(bucketKey);
    if (!table) return undefined;
    for (let len = Math.min(this.order, gram.length); len >= 0; len -= 1) {
      const key = gramKey(gram.slice(gram.length - len), len);
      const counts = table.get(key);
      if (counts && counts.size > 0) {
        return { counts, source };
      }
    }
    return undefined;
  }
}

export class NgramMovementBackend implements MovementModelBackend {
  readonly id = MOCK_MOVEMENT_BACKEND_ID;

  async train(
    dataset: MovementTrainingDataset,
    config: MovementModelTrainingConfig = {},
  ): Promise<TrainedMovementModel> {
    const order = Math.max(1, config.order ?? DEFAULT_ORDER);
    const transitions = new Map<string, GramTable>();
    const templates = new Map<string, Template>();

    for (const trajectory of dataset.trajectories) {
      learnTrajectory(trajectory, order, transitions, templates);
    }

    return new NgramMovementModel(order, 1, transitions, templates);
  }

  load(serialized: SerializedMovementModel): TrainedMovementModel {
    if (serialized.backendId !== this.id) {
      throw new Error(`cannot load model for backend ${serialized.backendId} into ${this.id}`);
    }
    const transitions = new Map<string, GramTable>();
    for (const [bucketKey, table] of Object.entries(serialized.transitions)) {
      const gramTable: GramTable = new Map();
      for (const [gram, counts] of Object.entries(table)) {
        gramTable.set(gram, new Map(Object.entries(counts)));
      }
      transitions.set(bucketKey, gramTable);
    }
    const templates = new Map<string, Template>(Object.entries(serialized.templates));
    return new NgramMovementModel(serialized.order, serialized.modelVersion, transitions, templates);
  }
}

function learnTrajectory(
  trajectory: MovementTrajectory,
  order: number,
  transitions: Map<string, GramTable>,
  templates: Map<string, Template>,
): void {
  const tokens: string[] = [];
  let prevTs: number | undefined;
  for (const event of trajectory.events) {
    const token = movementEventToken(event);
    tokens.push(token);
    if (!templates.has(token)) {
      const { ts, ...rest } = event;
      templates.set(token, { event: rest, dt: prevTs === undefined ? 0 : Math.max(1, ts - prevTs) });
    }
    prevTs = event.ts;
  }
  tokens.push(MOVEMENT_END_TOKEN);

  const bucketKeys = [
    movementContextKey(trajectory.context),
    movementAppKey(trajectory.context),
    MOVEMENT_GLOBAL_KEY,
  ];

  const history: string[] = [MOVEMENT_START_TOKEN];
  for (const next of tokens) {
    // Record every gram length 0..order so lookups can back off.
    for (let len = 0; len <= order; len += 1) {
      const key = gramKey(history.slice(history.length - len), len);
      for (const bucketKey of bucketKeys) {
        increment(transitions, bucketKey, key, next);
      }
    }
    history.push(next);
    if (history.length > order) {
      history.splice(0, history.length - order);
    }
  }
}

function increment(
  transitions: Map<string, GramTable>,
  bucketKey: string,
  gram: string,
  token: string,
): void {
  let table = transitions.get(bucketKey);
  if (!table) {
    table = new Map();
    transitions.set(bucketKey, table);
  }
  let counts = table.get(gram);
  if (!counts) {
    counts = new Map();
    table.set(gram, counts);
  }
  counts.set(token, (counts.get(token) ?? 0) + 1);
}

function prefixToGram(prefix: MovementEvent[], order: number): string[] {
  const tokens = prefix.map(movementEventToken);
  const gram = [MOVEMENT_START_TOKEN, ...tokens];
  return gram.slice(gram.length - order);
}

function gramKey(gram: string[], len: number): string {
  return `${len}|${gram.join(" ")}`;
}

/** Deterministic argmax: highest count wins, ties broken by lexical token order. */
function argmax(counts: CountMap): string | undefined {
  let best: string | undefined;
  let bestCount = -1;
  for (const [token, count] of counts) {
    if (count > bestCount || (count === bestCount && (best === undefined || token < best))) {
      best = token;
      bestCount = count;
    }
  }
  return best;
}

function confidenceOf(counts: CountMap, token: string): number {
  let total = 0;
  for (const count of counts.values()) total += count;
  if (total === 0) return 0;
  return (counts.get(token) ?? 0) / total;
}

function materialize(template: Template, lastTs: number): MovementEvent {
  return { ...template.event, ts: lastTs + Math.max(1, template.dt) };
}
