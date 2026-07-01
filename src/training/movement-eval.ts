import type {
  MovementEvent,
  MovementGesture,
  MovementModel,
  MovementTrajectory,
} from "./movement-policy.js";
import { rolloutMovements } from "./movement-policy.js";

/**
 * Synthetic movement-stream generator + generalization eval harness for the
 * movement-policy subsystem. Lets the capture→dataset→train→replay pipeline be
 * validated deterministically in the cloud with no real OS input (objective #2).
 */

type MovementStep = {
  gesture: MovementGesture;
  /** Pool of interchangeable targets — variation here yields related-but-new flows. */
  targets: string[];
  direction?: MovementEvent["direction"];
};

/** A parametric task: a fixed gesture *shape* with variable concrete targets. */
export type MovementTaskTemplate = {
  name: string;
  app: string;
  steps: MovementStep[];
};

export const DEFAULT_MOVEMENT_TEMPLATES: MovementTaskTemplate[] = [
  {
    name: "compose-email",
    app: "mail",
    steps: [
      { gesture: "tap", targets: ["compose-button", "new-message"] },
      { gesture: "type", targets: ["recipient-field"] },
      { gesture: "type", targets: ["subject-field"] },
      { gesture: "type", targets: ["body-field"] },
      { gesture: "tap", targets: ["send-button"] },
    ],
  },
  {
    name: "browse-scroll-open",
    app: "browser",
    steps: [
      { gesture: "tap", targets: ["address-bar"] },
      { gesture: "type", targets: ["url-field"] },
      { gesture: "scroll", targets: ["page"], direction: "down" },
      { gesture: "tap", targets: ["result-link", "headline-link"] },
    ],
  },
];

/**
 * Generate synthetic movement trajectories from templates. Different template
 * instances draw different concrete targets from each step's pool, producing
 * *related but distinct* flows — the substrate for measuring generalization.
 */
export function generateSyntheticMovementTrajectories(options: {
  templates?: MovementTaskTemplate[];
  perTemplate?: number;
  seed?: number;
  idPrefix?: string;
}): MovementTrajectory[] {
  const templates = options.templates ?? DEFAULT_MOVEMENT_TEMPLATES;
  const perTemplate = options.perTemplate ?? 8;
  const seed = options.seed ?? 1;
  const idPrefix = options.idPrefix ?? "syn";
  const trajectories: MovementTrajectory[] = [];

  for (const template of templates) {
    for (let instance = 0; instance < perTemplate; instance += 1) {
      const events: MovementEvent[] = template.steps.map((step, index) => {
        // Deterministic-but-varied: cycle targets by instance so multi-target
        // steps produce related-but-distinct flows reproducibly across seeds.
        const pick = (instance + index + seed) % step.targets.length;
        const target = step.targets[pick] ?? step.targets[0];
        return {
          app: template.app,
          gesture: step.gesture,
          ...(target ? { target } : {}),
          ...(step.direction ? { direction: step.direction } : {}),
          ts: index,
        };
      });
      trajectories.push({ id: `${idPrefix}-${template.name}-${instance}`, app: template.app, events });
    }
  }
  return trajectories;
}

/** Deterministic split into train/holdout, preserving template coverage. */
export function splitTrajectories(
  trajectories: MovementTrajectory[],
  holdoutRatio = 0.25,
): { train: MovementTrajectory[]; holdout: MovementTrajectory[] } {
  const train: MovementTrajectory[] = [];
  const holdout: MovementTrajectory[] = [];
  const stride = Math.max(2, Math.round(1 / Math.min(Math.max(holdoutRatio, 0.01), 0.99)));
  trajectories.forEach((trajectory, index) => {
    if (index % stride === stride - 1) {
      holdout.push(trajectory);
    } else {
      train.push(trajectory);
    }
  });
  return { train, holdout };
}

export type MovementEvalMetrics = {
  trajectoryCount: number;
  stepCount: number;
  /** Teacher-forced next-token top-1 accuracy (exact gesture+target+direction). */
  tokenAccuracy: number;
  /** Teacher-forced next-gesture top-1 accuracy (movement *shape*). */
  gestureAccuracy: number;
  /** Fraction of trajectories whose free rollout exactly reproduces the record. */
  rolloutExactMatch: number;
};

function sameToken(a: MovementEvent | undefined, b: MovementEvent): boolean {
  return (
    a !== undefined &&
    a.gesture === b.gesture &&
    (a.target ?? "") === (b.target ?? "") &&
    (a.direction ?? "") === (b.direction ?? "")
  );
}

/**
 * Evaluate a trained model against a set of trajectories.
 *
 * On the *training* set this measures replay fidelity (piece c). On a *holdout*
 * of related-but-unseen flows it measures generalization (piece d): even when
 * exact tokens are novel, a generalizing policy keeps a high `gestureAccuracy`.
 */
export function evaluateMovementModel(
  model: MovementModel,
  trajectories: MovementTrajectory[],
): MovementEvalMetrics {
  let steps = 0;
  let tokenHits = 0;
  let gestureHits = 0;
  let rolloutMatches = 0;

  for (const trajectory of trajectories) {
    for (let i = 0; i < trajectory.events.length; i += 1) {
      const expected = trajectory.events[i]!;
      const prediction = model.predict({ app: trajectory.app, history: trajectory.events.slice(0, i) });
      steps += 1;
      if (sameToken(prediction.event, expected)) {
        tokenHits += 1;
      }
      if (prediction.gesture === expected.gesture) {
        gestureHits += 1;
      }
    }

    const rollout = rolloutMovements(model, { app: trajectory.app }, { maxSteps: trajectory.events.length + 4 });
    if (
      rollout.length === trajectory.events.length &&
      rollout.every((event, index) => sameToken(event, trajectory.events[index]!))
    ) {
      rolloutMatches += 1;
    }
  }

  return {
    trajectoryCount: trajectories.length,
    stepCount: steps,
    tokenAccuracy: steps === 0 ? 0 : tokenHits / steps,
    gestureAccuracy: steps === 0 ? 0 : gestureHits / steps,
    rolloutExactMatch: trajectories.length === 0 ? 0 : rolloutMatches / trajectories.length,
  };
}
