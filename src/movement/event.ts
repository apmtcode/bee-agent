import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: atomic event schema.
 *
 * A `MovementEvent` is the smallest replayable unit the movement subsystem
 * learns from — a single mouse/keyboard/gesture action on the local computer.
 * The recorder/adapters (`src/capture`) produce high-level trajectory actions;
 * this module normalizes them into a compact, tokenizable stream that a local
 * model can be post-trained on to repeat and generalize movements.
 *
 * Everything here is deterministic and OS-free so the pipeline can be validated
 * in the cloud with synthetic streams; the real on-device recording feeds the
 * same schema when bee-agent runs locally.
 */

export const MOVEMENT_EVENT_TYPES = [
  "move",
  "click",
  "drag",
  "scroll",
  "key",
  "type",
  "wait",
] as const;

export type MovementEventType = (typeof MOVEMENT_EVENT_TYPES)[number];

export type MovementButton = "left" | "right" | "middle";
export type MovementDirection = "up" | "down" | "left" | "right";

export type MovementEvent = {
  /** Milliseconds from the start of the owning sequence. */
  t: number;
  type: MovementEventType;
  /** Absolute pointer coordinates (screen or app-relative), when meaningful. */
  x?: number;
  y?: number;
  /** Relative displacement for move/drag/scroll deltas. */
  dx?: number;
  dy?: number;
  button?: MovementButton;
  key?: string;
  /** Semantic UI target ("submit-button", "search-field"). */
  target?: string;
  appId?: string;
};

export type MovementSequence = {
  id: string;
  appId?: string;
  /** Optional human/semantic label for the workflow this sequence performs. */
  label?: string;
  events: MovementEvent[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type TokenizeOptions = {
  /**
   * Grid size (px) used to quantize absolute coordinates when no direction is
   * available. Coarser grids generalize more; finer grids reproduce more
   * precisely. Defaults to 64.
   */
  gridSize?: number;
};

const DEFAULT_GRID_SIZE = 64;

/**
 * Token that marks the start of a sequence for n-gram context.
 */
export const MOVEMENT_START_TOKEN = "^";

/**
 * Map a single event onto a discrete token capturing its salient features.
 * Continuous coordinates are quantized so that spatially-similar movements
 * collapse to the same token — the mechanism by which the model generalizes.
 */
export function tokenizeEvent(event: MovementEvent, options: TokenizeOptions = {}): string {
  const gridSize = options.gridSize ?? DEFAULT_GRID_SIZE;
  const parts: string[] = [event.type];

  switch (event.type) {
    case "click":
    case "drag":
      if (event.button) {
        parts.push(event.button);
      }
      break;
    case "key":
    case "type":
      if (event.key) {
        parts.push(event.key);
      }
      break;
    case "scroll": {
      const direction = deltaDirection(event.dx, event.dy);
      if (direction) {
        parts.push(direction);
      }
      break;
    }
    case "move": {
      const direction = deltaDirection(event.dx, event.dy);
      if (direction) {
        parts.push(direction);
      }
      break;
    }
    case "wait":
      break;
  }

  if (event.target) {
    parts.push(`@${event.target}`);
  } else if ((event.type === "move" || event.type === "drag") && event.x !== undefined && event.y !== undefined) {
    // Fall back to a coarse grid cell so untargeted pointer moves still
    // tokenize to a repeatable, generalizable bucket.
    parts.push(`#${quantize(event.x, gridSize)},${quantize(event.y, gridSize)}`);
  }

  return parts.join(":");
}

export function tokenizeSequence(sequence: MovementSequence, options?: TokenizeOptions): string[] {
  return sequence.events.map((event) => tokenizeEvent(event, options));
}

function deltaDirection(dx?: number, dy?: number): MovementDirection | undefined {
  const x = dx ?? 0;
  const y = dy ?? 0;
  if (x === 0 && y === 0) {
    return undefined;
  }
  if (Math.abs(x) >= Math.abs(y)) {
    return x >= 0 ? "right" : "left";
  }
  return y >= 0 ? "down" : "up";
}

function quantize(value: number, gridSize: number): number {
  return Math.round(value / gridSize);
}

const GESTURE_TO_EVENT_TYPE: Record<string, MovementEventType> = {
  tap: "click",
  swipe: "drag",
  scroll: "scroll",
  type: "type",
  shortcut: "key",
};

/**
 * Derive a movement sequence from a captured trajectory span. Prefers the
 * reviewed/redacted actions (privacy-safe) when a review is present, so the
 * training path never sees raw capture. Timestamps are made relative to the
 * first action so sequences are position-independent in time.
 */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const reviewedActions = trajectory.review?.redactedActions;
  const rawActions = trajectory.actions;

  const sources = reviewedActions
    ? reviewedActions.map((action) => ({ ts: action.ts, summary: action.summary, metadata: undefined as Record<string, unknown> | undefined }))
    : rawActions.map((action) => ({ ts: action.ts, summary: action.summary, metadata: action.metadata }));

  const sorted = [...sources].sort((a, b) => a.ts - b.ts);
  const baseTs = sorted[0]?.ts ?? 0;

  const events: MovementEvent[] = sorted.map((action) => {
    const metadata = action.metadata ?? {};
    const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
    const mappedType = gesture ? GESTURE_TO_EVENT_TYPE[gesture] : undefined;
    const type: MovementEventType = mappedType ?? inferTypeFromSummary(action.summary);
    const direction = typeof metadata.direction === "string" ? (metadata.direction as MovementDirection) : undefined;
    const target = typeof metadata.target === "string" ? metadata.target : undefined;

    const event: MovementEvent = {
      t: action.ts - baseTs,
      type,
    };
    if (target) {
      event.target = target;
    }
    if (direction) {
      const { dx, dy } = directionToDelta(direction);
      event.dx = dx;
      event.dy = dy;
    }
    return event;
  });

  return {
    id: trajectory.id,
    events,
  };
}

function inferTypeFromSummary(summary: string): MovementEventType {
  const normalized = summary.toLowerCase();
  if (normalized.includes("scroll")) {
    return "scroll";
  }
  if (normalized.includes("swipe")) {
    return "drag";
  }
  if (normalized.includes("typed") || normalized.includes("type")) {
    return "type";
  }
  if (normalized.includes("shortcut") || normalized.includes("triggered")) {
    return "key";
  }
  return "click";
}

function directionToDelta(direction: MovementDirection): { dx: number; dy: number } {
  switch (direction) {
    case "up":
      return { dx: 0, dy: -1 };
    case "down":
      return { dx: 0, dy: 1 };
    case "left":
      return { dx: -1, dy: 0 };
    case "right":
      return { dx: 1, dy: 0 };
  }
}

/**
 * Small, fast, deterministic PRNG (mulberry32). Used only by the synthetic
 * generator so tests are reproducible without `Math.random`.
 */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SyntheticStreamOptions = {
  seed?: number;
  /** Number of sequences to generate. */
  count?: number;
  /** Workflow templates to sample from. Defaults to the built-in library. */
  templates?: MovementTemplate[];
  /** Positional jitter (px) applied to targeted coordinates for variety. */
  jitter?: number;
};

export type MovementTemplate = {
  label: string;
  appId: string;
  build: (rng: () => number, jitter: number) => MovementEvent[];
};

/**
 * A small library of realistic desktop workflow templates. Each produces a
 * fresh, slightly-varied event stream on every call so the generator can emit
 * many related-but-distinct sequences — exactly the distribution needed to
 * exercise "repeat" vs. "generalize".
 */
export const DEFAULT_MOVEMENT_TEMPLATES: MovementTemplate[] = [
  {
    label: "form-fill",
    appId: "browser",
    build: (rng, jitter) => {
      const j = () => Math.round((rng() - 0.5) * 2 * jitter);
      return [
        { t: 0, type: "move", x: 400 + j(), y: 200 + j(), target: "name-field" },
        { t: 120, type: "click", button: "left", target: "name-field" },
        { t: 300, type: "type", key: "text" },
        { t: 900, type: "move", x: 400 + j(), y: 280 + j(), target: "email-field" },
        { t: 1020, type: "click", button: "left", target: "email-field" },
        { t: 1200, type: "type", key: "text" },
        { t: 1800, type: "move", x: 500 + j(), y: 360 + j(), target: "submit-button" },
        { t: 1920, type: "click", button: "left", target: "submit-button" },
      ];
    },
  },
  {
    label: "scroll-read",
    appId: "reader",
    build: (rng, jitter) => {
      const j = () => Math.round((rng() - 0.5) * 2 * jitter);
      const events: MovementEvent[] = [
        { t: 0, type: "move", x: 600 + j(), y: 400 + j() },
      ];
      let t = 200;
      const scrolls = 3 + Math.floor(rng() * 3);
      for (let i = 0; i < scrolls; i += 1) {
        events.push({ t, type: "scroll", dy: 120, target: "article" });
        t += 350;
      }
      events.push({ t, type: "click", button: "left", target: "next-link" });
      return events;
    },
  },
  {
    label: "drag-reorder",
    appId: "board",
    build: (rng, jitter) => {
      const j = () => Math.round((rng() - 0.5) * 2 * jitter);
      return [
        { t: 0, type: "move", x: 300 + j(), y: 300 + j(), target: "card-a" },
        { t: 100, type: "drag", button: "left", dx: 0, dy: 200, target: "card-a" },
        { t: 600, type: "click", button: "left", target: "card-a" },
        { t: 800, type: "key", key: "Enter" },
      ];
    },
  },
];

/**
 * Generate a deterministic dataset of synthetic movement sequences. Because the
 * PRNG is seeded, the same seed always yields the same dataset — ideal for
 * repeatable train/eval round-trip tests in the cloud.
 */
export function generateSyntheticDataset(options: SyntheticStreamOptions = {}): MovementDataset {
  const seed = options.seed ?? 1;
  const count = options.count ?? 12;
  const templates = options.templates ?? DEFAULT_MOVEMENT_TEMPLATES;
  const jitter = options.jitter ?? 8;
  const rng = createSeededRandom(seed);

  const sequences: MovementSequence[] = [];
  for (let i = 0; i < count; i += 1) {
    const template = templates[Math.floor(rng() * templates.length)] ?? templates[0];
    if (!template) {
      break;
    }
    sequences.push({
      id: `${template.label}-${i}`,
      appId: template.appId,
      label: template.label,
      events: template.build(rng, jitter),
    });
  }

  return { version: 1, sequences };
}
