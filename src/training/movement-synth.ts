import type {
  CaptureTier,
  TrajectoryAction,
  TrajectoryObservation,
  TrajectorySpan,
} from "../capture/trajectory.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

/**
 * Deterministic synthetic movement-stream generator (standing objective #2's
 * "synthetic event-stream generator"). Because the engine runs in the cloud
 * with no access to a real machine, this fabricates realistic recorded
 * trajectories — sequences of observations + input actions — from seeded
 * randomness, so the whole capture→dataset→train→generalize→replay pipeline can
 * be validated end-to-end without real OS input capture.
 *
 * A "task template" is a labelled chain of gestures inside an app. The
 * generator emits many trajectories per template with jittered timing and
 * occasional variation, plus optional held-out variants (same task, unseen
 * surface details) to exercise generalization rather than memorization.
 */

export type MovementTaskTemplate = {
  /** Human-readable task label, e.g. "compose-and-send". */
  id: string;
  app: string;
  platform: string;
  /** Ordered gesture chain the operator performs to complete the task. */
  steps: Array<{
    tool: string;
    gesture?: string;
    target?: string;
    direction?: string;
    summary: string;
  }>;
  outcome?: "success" | "failure" | "aborted";
};

export type SynthesizeMovementOptions = {
  seed?: number;
  /** How many trajectories to emit per template. */
  perTemplate?: number;
  captureTier?: CaptureTier;
  /** Base timestamp; each event advances by a jittered delta from here. */
  startTs?: number;
  /**
   * With this probability [0,1] a step's target is swapped for a related
   * variant, producing "new but related" movements for generalization tests.
   */
  variationRate?: number;
};

/**
 * Small deterministic PRNG (mulberry32). Seeded so synthetic datasets — and any
 * test asserting on them — are fully reproducible across runs and machines.
 */
export class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  int(minInclusive: number, maxInclusive: number): number {
    return minInclusive + Math.floor(this.next() * (maxInclusive - minInclusive + 1));
  }

  chance(probability: number): boolean {
    return this.next() < probability;
  }

  pick<T>(items: readonly T[]): T {
    return items[this.int(0, items.length - 1)]!;
  }
}

/** A couple of default templates so callers can get a dataset with zero setup. */
export function defaultMovementTemplates(): MovementTaskTemplate[] {
  return [
    {
      id: "compose-and-send",
      app: "mail",
      platform: "macos",
      steps: [
        { tool: "device", gesture: "tap", target: "compose-button", summary: "tapped compose-button" },
        { tool: "device", gesture: "type", target: "recipient-field", summary: "typed into recipient-field" },
        { tool: "device", gesture: "type", target: "body-field", summary: "typed into body-field" },
        { tool: "device", gesture: "tap", target: "send-button", summary: "tapped send-button" },
      ],
      outcome: "success",
    },
    {
      id: "search-and-open",
      app: "browser",
      platform: "macos",
      steps: [
        { tool: "device", gesture: "shortcut", target: "focus-address-bar", summary: "triggered focus-address-bar" },
        { tool: "device", gesture: "type", target: "address-bar", summary: "typed into address-bar" },
        { tool: "device", gesture: "tap", target: "first-result", summary: "tapped first-result" },
        { tool: "device", gesture: "scroll", direction: "down", summary: "scrolled down" },
      ],
      outcome: "success",
    },
  ];
}

/** Related-target substitutions, keyed by the canonical target. */
const TARGET_VARIANTS: Record<string, string[]> = {
  "compose-button": ["new-message-button", "pencil-button"],
  "recipient-field": ["to-field", "address-field"],
  "body-field": ["message-field", "content-field"],
  "send-button": ["deliver-button", "submit-button"],
  "first-result": ["top-result", "result-1"],
  "address-bar": ["url-bar", "omnibox"],
};

export function synthesizeMovementTrajectories(
  templates: MovementTaskTemplate[],
  options: SynthesizeMovementOptions = {},
): TrajectorySpan[] {
  const rng = new SeededRandom(options.seed ?? 1);
  const perTemplate = options.perTemplate ?? 6;
  const captureTier = options.captureTier ?? "full";
  const variationRate = options.variationRate ?? 0;
  let ts = options.startTs ?? 1_000;

  const trajectories: TrajectorySpan[] = [];

  templates.forEach((template, templateIndex) => {
    for (let instance = 0; instance < perTemplate; instance += 1) {
      const observations: TrajectoryObservation[] = [];
      const actions: TrajectoryAction[] = [];

      // Opening observation establishes the app context for the whole chain.
      observations.push({
        kind: "observation",
        source: "device",
        summary: `${template.app} active on device`,
        ts,
        metadata: { appId: template.app, appName: template.app, platform: template.platform },
      });
      ts += rng.int(5, 40);

      for (const step of template.steps) {
        const target =
          step.target && variationRate > 0 && rng.chance(variationRate)
            ? rng.pick([step.target, ...(TARGET_VARIANTS[step.target] ?? [step.target])])
            : step.target;
        const summary = target ? step.summary.replace(step.target ?? "", target) : step.summary;
        actions.push({
          kind: "action",
          tool: step.tool,
          summary,
          ts,
          metadata: {
            ...(step.gesture ? { gesture: step.gesture } : {}),
            ...(target ? { target } : {}),
            ...(step.direction ? { direction: step.direction } : {}),
          },
        });
        ts += rng.int(10, 80);
      }

      trajectories.push(
        buildTrajectorySpanDeterministic({
          id: `synth-${template.id}-${instance}`,
          sessionId: `synth-session-${templateIndex}`,
          captureTier,
          observations,
          actions,
          outcome: { status: template.outcome ?? "success", summary: `${template.id} completed`, reward: 1 },
        }),
      );
    }
  });

  return trajectories;
}

/**
 * Like {@link buildTrajectorySpan} but with a caller-supplied createdAt so
 * synthetic output is byte-stable across runs (buildTrajectorySpan stamps
 * `new Date()`).
 */
function buildTrajectorySpanDeterministic(params: {
  id: string;
  sessionId: string;
  captureTier: CaptureTier;
  observations: TrajectoryObservation[];
  actions: TrajectoryAction[];
  outcome: TrajectorySpan["outcome"];
}): TrajectorySpan {
  const span = buildTrajectorySpan({
    id: params.id,
    sessionId: params.sessionId,
    captureTier: params.captureTier,
    observations: params.observations,
    actions: params.actions,
    outcome: params.outcome,
  });
  const firstTs = params.observations[0]?.ts ?? params.actions[0]?.ts ?? 0;
  return { ...span, createdAt: new Date(firstTs).toISOString() };
}
