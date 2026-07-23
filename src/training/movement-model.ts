import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning model.
 *
 * This module implements objective #2 parts (c) and (d) of the self-evolution
 * charter: an in-process model that can be *post-trained* on a recorded
 * movement dataset to (c) repeat the recorded movements and (d) generalize to
 * new but related movements.
 *
 * The model backend is deliberately pluggable ({@link MovementModelBackend}) so
 * that a real on-device small model (e.g. an MLX/torch policy) can be dropped in
 * for local execution, while the default {@link MarkovMovementBackend} is a
 * deterministic, dependency-free variable-order Markov model that runs anywhere
 * — including Anthropic's cloud, where there is no OS access. Everything here is
 * validated against synthetic/simulated event streams (see
 * {@link generateSyntheticMovementSequences}).
 */

/** A single atomic movement/action on the local computer. */
export type MovementEvent = {
  ts: number;
  /** Input surface, e.g. "mouse", "keyboard", "device", "window". */
  tool: string;
  /** Gesture kind, e.g. "click", "move", "scroll", "type", "shortcut", "tap". */
  gesture?: string;
  /** Normalized target label, e.g. "button:submit", "row:3". */
  target?: string;
  /** Movement direction where applicable. */
  direction?: string;
  /** Opaque value summary (never raw secrets — redacted upstream). */
  value?: string;
};

/** An ordered sequence of movements captured from one trajectory/session. */
export type MovementSequence = {
  id: string;
  events: MovementEvent[];
};

/** Canonical token string the model learns transitions over. */
export type MovementToken = string;

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainingConfig = {
  /** Maximum context order (number of preceding tokens conditioned on). */
  order?: number;
  /** Also learn target-family transitions so related-but-new targets generalize. */
  enableFamilyBackoff?: boolean;
};

/** Serializable, persistable trained-model artifact. */
export type MovementModelState = {
  backendId: string;
  version: 1;
  order: number;
  familyBackoff: boolean;
  vocabulary: MovementToken[];
  /** context (tokens joined by CTX_SEP) -> { nextToken: count } */
  transitions: Record<string, Record<MovementToken, number>>;
  /** target-family context -> { nextFamilyToken: count } */
  familyTransitions: Record<string, Record<MovementToken, number>>;
  trainedSequences: number;
  trainedEvents: number;
};

export type MovementPredictionSource = "exact" | "family";

export type MovementPrediction = {
  token: MovementToken;
  event: MovementEvent;
  confidence: number;
  /** Length of the context that actually matched (back-off order used). */
  backoffOrder: number;
  source: MovementPredictionSource;
};

/**
 * Pluggable backend seam. The deterministic {@link MarkovMovementBackend} is the
 * default; a real on-device model implements the same async contract (its
 * `train` shells out to a local trainer, `predict` to local inference) so the
 * rest of the pipeline is backend-agnostic.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<MovementModelState>;
  predict(state: MovementModelState, context: MovementToken[]): Promise<MovementPrediction | undefined>;
}

const FIELD_SEP = "␟"; // unit separator between token fields
const CTX_SEP = ""; // separator between tokens in a context key
const EMPTY = "-";
const DEFAULT_ORDER = 3;

function sanitizeField(value: string | undefined): string {
  if (value === undefined || value === "") {
    return EMPTY;
  }
  return value.trim().toLowerCase().replaceAll(FIELD_SEP, " ").replaceAll(CTX_SEP, " ");
}

/**
 * Collapse trailing instance identifiers so that `row:3` and `row:17`, or
 * `button:submit-1` and `button:submit-2`, share a family — the mechanism that
 * lets the model generalize a learned movement onto a new but related target.
 */
export function targetFamily(target: string | undefined): string {
  const base = sanitizeField(target);
  if (base === EMPTY) {
    return EMPTY;
  }
  return base
    .replace(/[:_\-#/ ]?\d+$/g, "")
    .replace(/[:_\-#/ ]?[0-9a-f]{6,}$/g, "")
    .trim() || base;
}

/** Encode a movement into its canonical token. */
export function tokenOf(event: MovementEvent): MovementToken {
  return [sanitizeField(event.tool), sanitizeField(event.gesture), sanitizeField(event.target), sanitizeField(event.direction)].join(
    FIELD_SEP,
  );
}

/** Encode a movement into its family token (target collapsed to its family). */
export function familyTokenOf(event: MovementEvent): MovementToken {
  return [sanitizeField(event.tool), sanitizeField(event.gesture), targetFamily(event.target), sanitizeField(event.direction)].join(
    FIELD_SEP,
  );
}

/** Family of an already-encoded exact token. */
export function familyOfToken(token: MovementToken): MovementToken {
  const [tool, gesture, target, direction] = token.split(FIELD_SEP);
  return [tool ?? EMPTY, gesture ?? EMPTY, targetFamily(target), direction ?? EMPTY].join(FIELD_SEP);
}

/** Decode a token back into a (partial) movement to perform. */
export function eventFromToken(token: MovementToken, ts = 0): MovementEvent {
  const [tool, gesture, target, direction] = token.split(FIELD_SEP);
  const event: MovementEvent = { ts, tool: tool ?? EMPTY };
  if (gesture && gesture !== EMPTY) {
    event.gesture = gesture;
  }
  if (target && target !== EMPTY) {
    event.target = target;
  }
  if (direction && direction !== EMPTY) {
    event.direction = direction;
  }
  return event;
}

function argmax(counts: Record<MovementToken, number>): { token: MovementToken; count: number; total: number } | undefined {
  let best: string | undefined;
  let bestCount = -1;
  let total = 0;
  // Deterministic: highest count wins, ties broken by lexicographic token order.
  for (const token of Object.keys(counts).sort()) {
    const count = counts[token] ?? 0;
    total += count;
    if (count > bestCount) {
      bestCount = count;
      best = token;
    }
  }
  if (best === undefined) {
    return undefined;
  }
  return { token: best, count: bestCount, total };
}

/**
 * Deterministic, dependency-free variable-order Markov model with target-family
 * back-off. Learns higher-order context → next-movement transitions for
 * high-fidelity replay, and target-family transitions so unseen-but-related
 * movements still resolve to a plausible next movement.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";

  async train(dataset: MovementDataset, config: MovementTrainingConfig = {}): Promise<MovementModelState> {
    const order = Math.max(1, config.order ?? DEFAULT_ORDER);
    const familyBackoff = config.enableFamilyBackoff ?? true;
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const familyTransitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let trainedEvents = 0;

    for (const sequence of dataset.sequences) {
      const tokens = sequence.events.map(tokenOf);
      const familyTokens = sequence.events.map(familyTokenOf);
      trainedEvents += tokens.length;
      for (const token of tokens) {
        vocabulary.add(token);
      }
      for (let i = 1; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        for (let k = 1; k <= order && k <= i; k += 1) {
          record(transitions, tokens.slice(i - k, i).join(CTX_SEP), next);
        }
        if (familyBackoff) {
          const nextFamily = familyTokens[i]!;
          for (let k = 1; k <= order && k <= i; k += 1) {
            record(familyTransitions, familyTokens.slice(i - k, i).join(CTX_SEP), nextFamily);
          }
        }
      }
    }

    return {
      backendId: this.id,
      version: 1,
      order,
      familyBackoff,
      vocabulary: [...vocabulary].sort(),
      transitions,
      familyTransitions,
      trainedSequences: dataset.sequences.length,
      trainedEvents,
    };
  }

  async predict(state: MovementModelState, context: MovementToken[]): Promise<MovementPrediction | undefined> {
    // Prefer the highest-order exact context that has been observed.
    for (let k = Math.min(state.order, context.length); k >= 1; k -= 1) {
      const key = context.slice(context.length - k).join(CTX_SEP);
      const counts = state.transitions[key];
      const choice = counts ? argmax(counts) : undefined;
      if (choice) {
        return {
          token: choice.token,
          event: eventFromToken(choice.token),
          confidence: choice.total > 0 ? choice.count / choice.total : 0,
          backoffOrder: k,
          source: "exact",
        };
      }
    }

    // Generalization: back off to target-family context for related movements.
    if (state.familyBackoff) {
      const familyContext = context.map(familyOfToken);
      for (let k = Math.min(state.order, familyContext.length); k >= 1; k -= 1) {
        const key = familyContext.slice(familyContext.length - k).join(CTX_SEP);
        const counts = state.familyTransitions[key];
        const choice = counts ? argmax(counts) : undefined;
        if (choice) {
          return {
            token: choice.token,
            event: eventFromToken(choice.token),
            confidence: choice.total > 0 ? choice.count / choice.total : 0,
            backoffOrder: k,
            source: "family",
          };
        }
      }
    }

    return undefined;
  }
}

function record(table: Record<string, Record<MovementToken, number>>, context: string, next: MovementToken): void {
  const bucket = (table[context] ??= {});
  bucket[next] = (bucket[next] ?? 0) + 1;
}

/**
 * High-level orchestrator that ties a dataset to a backend and offers the two
 * capabilities the charter asks for: replay (repeat recorded movements) and
 * evaluation of replay fidelity + generalization.
 */
export class MovementModelTrainer {
  constructor(private readonly backend: MovementModelBackend = new MarkovMovementBackend()) {}

  get backendId(): string {
    return this.backend.id;
  }

  async train(sequences: MovementSequence[], config?: MovementTrainingConfig): Promise<MovementModelState> {
    return await this.backend.train({ sequences }, config);
  }

  async predictNext(state: MovementModelState, context: MovementEvent[]): Promise<MovementPrediction | undefined> {
    return await this.backend.predict(state, context.map(tokenOf));
  }

  /**
   * Repeat/continue a movement: from a seed prefix, iteratively predict and
   * emit the next movement until `steps` movements are produced or the model has
   * nothing more to predict.
   */
  async replay(state: MovementModelState, seed: MovementEvent[], steps: number): Promise<MovementEvent[]> {
    const context = seed.map(tokenOf);
    const produced: MovementEvent[] = [];
    for (let i = 0; i < steps; i += 1) {
      const prediction = await this.backend.predict(state, context);
      if (!prediction) {
        break;
      }
      produced.push({ ...prediction.event, ts: seed.length + i });
      context.push(prediction.token);
    }
    return produced;
  }

  /**
   * Replay fidelity on a (typically seen) sequence: for each position, predict
   * the next movement from the true prefix and score an exact token match. On
   * trained sequences with distinctive contexts this approaches 1.0 — i.e. the
   * model can repeat the recorded movements.
   */
  async evaluateReplayFidelity(state: MovementModelState, sequence: MovementSequence): Promise<MovementFidelityReport> {
    const tokens = sequence.events.map(tokenOf);
    let scored = 0;
    let exact = 0;
    let predicted = 0;
    for (let i = 1; i < tokens.length; i += 1) {
      scored += 1;
      const prediction = await this.backend.predict(state, tokens.slice(0, i));
      if (!prediction) {
        continue;
      }
      predicted += 1;
      if (prediction.token === tokens[i]) {
        exact += 1;
      }
    }
    return {
      scored,
      predicted,
      exact,
      exactRate: scored > 0 ? exact / scored : 0,
      coverage: scored > 0 ? predicted / scored : 0,
    };
  }

  /**
   * Generalization on a held-out but related sequence: score both exact token
   * match and (looser) target-family match, since a generalized movement is
   * correct at the family level even when the specific target instance is new.
   */
  async evaluateGeneralization(state: MovementModelState, sequence: MovementSequence): Promise<MovementGeneralizationReport> {
    const tokens = sequence.events.map(tokenOf);
    let scored = 0;
    let exact = 0;
    let familyMatch = 0;
    let predicted = 0;
    for (let i = 1; i < tokens.length; i += 1) {
      scored += 1;
      const prediction = await this.backend.predict(state, tokens.slice(0, i));
      if (!prediction) {
        continue;
      }
      predicted += 1;
      if (prediction.token === tokens[i]) {
        exact += 1;
      }
      if (familyOfToken(prediction.token) === familyOfToken(tokens[i]!)) {
        familyMatch += 1;
      }
    }
    return {
      scored,
      predicted,
      exact,
      familyMatch,
      exactRate: scored > 0 ? exact / scored : 0,
      familyRate: scored > 0 ? familyMatch / scored : 0,
      coverage: scored > 0 ? predicted / scored : 0,
    };
  }
}

export type MovementFidelityReport = {
  scored: number;
  predicted: number;
  exact: number;
  exactRate: number;
  coverage: number;
};

export type MovementGeneralizationReport = MovementFidelityReport & {
  familyMatch: number;
  familyRate: number;
};

/** Derive a movement from a captured trajectory action. */
export function movementFromAction(action: TrajectoryAction): MovementEvent {
  const metadata = action.metadata ?? {};
  const gesture = pickString(metadata, "gesture");
  const target = pickString(metadata, "target") ?? pickString(metadata, "selector");
  const direction = pickString(metadata, "direction");
  const value = pickString(metadata, "valueSummary") ?? pickString(metadata, "value");
  const event: MovementEvent = { ts: action.ts, tool: action.tool };
  if (gesture) {
    event.gesture = gesture;
  }
  if (target) {
    event.target = target;
  }
  if (direction) {
    event.direction = direction;
  }
  if (value) {
    event.value = value;
  }
  return event;
}

/** Derive a movement sequence from a trajectory span's ordered actions. */
export function deriveMovementSequence(trajectory: TrajectorySpan): MovementSequence {
  const events = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map(movementFromAction);
  return { id: trajectory.id, events };
}

function pickString(metadata: Record<string, unknown>, key: string): string | undefined {
  const value = metadata[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/**
 * Deterministic synthetic event-stream generator for validating the
 * capture→dataset→train→replay pipeline without real OS input. Produces
 * `families` movement "recipes"; each recipe yields `variantsPerFamily`
 * sequences that share the same gesture/target-family structure but differ in
 * concrete target instances — exactly the "new but related movement" shape the
 * generalization eval needs.
 */
export function generateSyntheticMovementSequences(params: {
  families?: number;
  variantsPerFamily?: number;
  stepsPerSequence?: number;
  seed?: number;
}): MovementSequence[] {
  const families = Math.max(1, params.families ?? 3);
  const variants = Math.max(1, params.variantsPerFamily ?? 3);
  const steps = Math.max(2, params.stepsPerSequence ?? 5);
  let state = (params.seed ?? 1) >>> 0 || 1;
  // Deterministic LCG so runs are reproducible without Math.random.
  const next = (): number => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };

  const tools = ["mouse", "keyboard", "window"];
  const gestures = ["move", "click", "scroll", "type", "shortcut"];
  const sequences: MovementSequence[] = [];

  for (let f = 0; f < families; f += 1) {
    // A recipe fixes the tool/gesture/target-family per step; variants only
    // change the concrete target instance index.
    const recipe = Array.from({ length: steps }, (_, step) => ({
      tool: tools[Math.floor(next() * tools.length)] ?? "mouse",
      gesture: gestures[Math.floor(next() * gestures.length)] ?? "click",
      targetFamily: `panel${f}:item`,
      direction: step % 2 === 0 ? undefined : (next() > 0.5 ? "down" : "up"),
    }));

    for (let v = 0; v < variants; v += 1) {
      const events: MovementEvent[] = recipe.map((step, index) => ({
        ts: index,
        tool: step.tool,
        gesture: step.gesture,
        target: `${step.targetFamily}-${v * steps + index}`,
        ...(step.direction ? { direction: step.direction } : {}),
      }));
      sequences.push({ id: `family${f}-variant${v}`, events });
    }
  }

  return sequences;
}
