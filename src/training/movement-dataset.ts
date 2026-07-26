import type { TrajectoryAction, TrajectoryObservation, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-learning dataset format.
 *
 * The local-movement learning subsystem needs a structured, replayable dataset
 * of the low-level movements a user performs (pointer / keyboard / gesture /
 * window / command events). Raw capture produces {@link TrajectorySpan}s whose
 * observations and actions carry rich metadata; this module distills those into
 * canonical {@link MovementToken}s and groups them into ordered
 * {@link MovementSequence}s that a local model can train on and replay.
 *
 * Everything here is pure and deterministic so it runs in the cloud / CI without
 * any real OS input — the on-device capture feeds the same schema when bee-agent
 * runs locally.
 */

export const MOVEMENT_MODALITIES = [
  "pointer",
  "keyboard",
  "gesture",
  "window",
  "command",
] as const;

export type MovementModality = (typeof MOVEMENT_MODALITIES)[number];

/**
 * A single canonical movement. `modality` + `verb` + `target` (+ optional
 * `detail`) fully describe one replayable step. Targets are normalized so that
 * semantically-equal targets collapse to the same token, which is what lets a
 * model generalize across related-but-not-identical movement streams.
 */
export type MovementToken = {
  modality: MovementModality;
  verb: string;
  target: string;
  detail?: string;
};

export type MovementSequence = {
  id: string;
  /** Optional family/task label — used to group related sequences for eval. */
  label?: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

const WILDCARD = "*";
const KEY_SEP = "|";

/** Normalize a free-form target into a stable, collision-free token segment. */
export function normalizeMovementTarget(raw: string | undefined | null): string {
  if (typeof raw !== "string") {
    return WILDCARD;
  }
  const trimmed = raw.trim().toLowerCase();
  if (!trimmed) {
    return WILDCARD;
  }
  const slug = trimmed
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9._/-]/g, "")
    .replace(/^[-.]+|[-.]+$/g, "");
  return slug || WILDCARD;
}

/** Canonical, delimiter-safe string key for a token (used as the model vocab). */
export function movementTokenKey(token: MovementToken): string {
  return [token.modality, token.verb, token.target, token.detail ?? ""].join(KEY_SEP);
}

/** Reconstruct a token from its canonical key (inverse of {@link movementTokenKey}). */
export function movementTokenFromKey(key: string): MovementToken {
  const [modality, verb, target, detail] = key.split(KEY_SEP);
  return {
    modality: (MOVEMENT_MODALITIES as readonly string[]).includes(modality ?? "")
      ? (modality as MovementModality)
      : "command",
    verb: verb ?? "",
    target: target ?? WILDCARD,
    ...(detail ? { detail } : {}),
  };
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

/** Map one captured action into a movement token (undefined = not a movement). */
export function actionToMovementToken(action: TrajectoryAction): MovementToken | undefined {
  const meta = action.metadata ?? {};
  const gesture = readString(meta.gesture);
  if (gesture) {
    switch (gesture) {
      case "tap":
        return { modality: "pointer", verb: "tap", target: normalizeMovementTarget(readString(meta.target)) };
      case "swipe":
        return {
          modality: "pointer",
          verb: "swipe",
          target: WILDCARD,
          detail: normalizeMovementTarget(readString(meta.direction)),
        };
      case "scroll":
        return {
          modality: "pointer",
          verb: "scroll",
          target: WILDCARD,
          detail: normalizeMovementTarget(readString(meta.direction)),
        };
      case "type":
        return { modality: "keyboard", verb: "type", target: normalizeMovementTarget(readString(meta.target)) };
      case "shortcut":
        return { modality: "keyboard", verb: "shortcut", target: normalizeMovementTarget(readString(meta.target)) };
      default:
        return { modality: "gesture", verb: normalizeMovementTarget(gesture), target: normalizeMovementTarget(readString(meta.target)) };
    }
  }
  return { modality: "command", verb: normalizeMovementTarget(action.tool), target: WILDCARD };
}

/** Map one captured observation into a movement token (undefined = not a movement). */
export function observationToMovementToken(observation: TrajectoryObservation): MovementToken | undefined {
  const meta = observation.metadata ?? {};
  const event = readString(meta.event);
  switch (event) {
    case "focus-changed":
      return { modality: "window", verb: "focus", target: normalizeMovementTarget(readString(meta.windowTitle)) };
    case "window-opened":
      return { modality: "window", verb: "open", target: normalizeMovementTarget(readString(meta.windowTitle)) };
    case "file-opened":
      return { modality: "command", verb: "open", target: normalizeMovementTarget(readString(meta.filePath)) };
    case "command-ran":
      return { modality: "command", verb: "run", target: normalizeMovementTarget(readString(meta.commandSummary)) };
    default:
      return undefined;
  }
}

/**
 * Distill a trajectory span into an ordered movement sequence. Observations and
 * actions are merged and ordered by timestamp (actions break ties after
 * observations, matching the replay-manifest ordering convention).
 */
export function tokenizeTrajectory(span: TrajectorySpan): MovementSequence {
  const events: Array<{ ts: number; order: number; token: MovementToken }> = [];
  for (const observation of span.observations) {
    const token = observationToMovementToken(observation);
    if (token) {
      events.push({ ts: observation.ts, order: 0, token });
    }
  }
  for (const action of span.actions) {
    const token = actionToMovementToken(action);
    if (token) {
      events.push({ ts: action.ts, order: 1, token });
    }
  }
  events.sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : a.order - b.order));
  return { id: span.id, tokens: events.map((event) => event.token) };
}

/** Build a movement dataset from captured spans, dropping empty sequences. */
export function buildMovementDataset(spans: TrajectorySpan[]): MovementDataset {
  return {
    version: 1,
    sequences: spans.map(tokenizeTrajectory).filter((sequence) => sequence.tokens.length > 0),
  };
}
