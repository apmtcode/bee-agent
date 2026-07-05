import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * The local-movement learning subsystem operates over a *discretized* token
 * stream rather than raw pixel coordinates. A `MovementToken` is the canonical,
 * comparable unit a model learns to predict: an action verb plus an optional
 * descriptor (target / direction / value bucket), e.g. `tap:submit`,
 * `swipe:down`, `type:search-field`.
 *
 * Discretizing here (rather than in the model) keeps the backend interface
 * simple and makes datasets human-reviewable, which the capture consent flow
 * requires. Backends stay pluggable: they consume `MovementSequence`s and never
 * see the raw `TrajectorySpan`.
 */
export type MovementToken = string;

/** Reserved boundary tokens. Real movement tokens never collide with these. */
export const MOVEMENT_START_TOKEN: MovementToken = "START";
export const MOVEMENT_END_TOKEN: MovementToken = "END";

export type MovementEvent = {
  ts: number;
  /** Canonical discrete token, e.g. `tap:submit`. */
  token: MovementToken;
  /** The action verb (gesture kind or tool name), e.g. `tap`, `type`, `swipe`. */
  action: string;
  /** Optional descriptor the action operated on (UI target, direction, bucket). */
  descriptor?: string;
};

export type MovementSequence = {
  id: string;
  sessionId: string;
  events: MovementEvent[];
  outcome?: "success" | "failure" | "aborted";
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

/**
 * Collapse an arbitrary free-text label into a stable, comparable slug so that
 * "Submit Order" and "submit order" tokenize identically. Kept deliberately
 * lossy — the goal is generalization across near-identical UI affordances.
 */
export function slugifyMovementLabel(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

/**
 * Build the canonical token for an action verb + optional descriptor. The verb
 * itself is slugified so `Tap` and `tap` collapse together.
 */
export function buildMovementToken(action: string, descriptor?: string): MovementToken {
  const verb = slugifyMovementLabel(action) || "act";
  const detail = descriptor ? slugifyMovementLabel(descriptor) : "";
  return detail ? `${verb}:${detail}` : verb;
}

/**
 * Derive the descriptor for a trajectory action, preferring structured metadata
 * (target / direction / value summary) and falling back to the human summary.
 */
function deriveActionDescriptor(action: TrajectoryAction): string | undefined {
  const metadata = action.metadata ?? {};
  const target = typeof metadata.target === "string" ? metadata.target : undefined;
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  const valueSummary = typeof metadata.valueSummary === "string" ? metadata.valueSummary : undefined;
  return target ?? direction ?? valueSummary;
}

/**
 * Derive the action verb for a trajectory action, preferring an explicit
 * `gesture` metadata kind (tap/swipe/…) and otherwise the tool name.
 */
function deriveActionVerb(action: TrajectoryAction): string {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  return gesture ?? action.tool;
}

/**
 * Convert a captured/reviewed trajectory span into a discretized movement
 * sequence. Observations are intentionally dropped — the model learns the
 * *movement policy* (what to do next), not the observation stream.
 */
export function tokenizeTrajectorySpan(span: TrajectorySpan): MovementSequence {
  const events = [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map<MovementEvent>((action) => {
      const verb = deriveActionVerb(action);
      const descriptor = deriveActionDescriptor(action);
      return {
        ts: action.ts,
        token: buildMovementToken(verb, descriptor),
        action: slugifyMovementLabel(verb) || "act",
        ...(descriptor ? { descriptor: slugifyMovementLabel(descriptor) } : {}),
      };
    });

  return {
    id: span.id,
    sessionId: span.sessionId,
    events,
    ...(span.outcome ? { outcome: span.outcome.status } : {}),
  };
}

export function tokenizeTrajectoryDataset(spans: TrajectorySpan[]): MovementDataset {
  return { sequences: spans.map(tokenizeTrajectorySpan) };
}

/** Extract just the token list from a sequence (convenience for backends). */
export function sequenceTokens(sequence: MovementSequence): MovementToken[] {
  return sequence.events.map((event) => event.token);
}
