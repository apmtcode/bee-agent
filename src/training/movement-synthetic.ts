// Deterministic synthetic movement-stream generator.
//
// bee-agent runs in the cloud with no access to a real mouse/keyboard, so we
// synthesize structured, repeatable movement trajectories to validate the
// capture -> dataset -> train -> replay -> generalize loop. Everything here is
// a pure function of its inputs (no clock, no randomness) so tests are stable.

import type {
  MovementContext,
  MovementEvent,
  MovementTrainingDataset,
  MovementTrajectory,
} from "./movement-model.js";

export type SyntheticTaskSpec = {
  id: string;
  context: MovementContext;
  /** Ordered UI targets the task moves through (e.g. buttons, fields). */
  targets: string[];
  /** Optional text typed after reaching the final target. */
  typeValue?: string;
};

const STEP_DT = 16;

/**
 * Build a deterministic "move to each target, click it, optionally type"
 * trajectory. Coordinates are derived from the target index so related tasks
 * that share targets produce overlapping tokens (which is what the model
 * generalizes over).
 */
export function synthesizeTrajectory(spec: SyntheticTaskSpec, startTs = 0): MovementTrajectory {
  const events: MovementEvent[] = [];
  let ts = startTs;
  const push = (event: Omit<MovementEvent, "ts">): void => {
    events.push({ ...event, ts });
    ts += STEP_DT;
  };

  spec.targets.forEach((target, index) => {
    const x = 40 + index * 120;
    const y = 60 + index * 40;
    push({ actor: "mouse", action: "move", target, x, y });
    push({ actor: "mouse", action: "click", target, x, y });
  });

  if (spec.typeValue !== undefined) {
    push({ actor: "keyboard", action: "type", value: spec.typeValue });
    push({ actor: "keyboard", action: "keydown", key: "Enter" });
  }

  return { id: spec.id, context: spec.context, events };
}

export function synthesizeDataset(specs: SyntheticTaskSpec[], startTs = 0): MovementTrainingDataset {
  let ts = startTs;
  const trajectories = specs.map((spec) => {
    const trajectory = synthesizeTrajectory(spec, ts);
    const last = trajectory.events[trajectory.events.length - 1];
    ts = (last ? last.ts : ts) + 1000;
    return trajectory;
  });
  return { trajectories };
}

/**
 * Compare two event streams by their semantic fields (ignoring absolute ts).
 * Used to assert replay fidelity — that the model repeats the recorded
 * movements, not their exact wall-clock timing.
 */
export function movementsMatch(a: MovementEvent[], b: MovementEvent[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((event, index) => {
    const other = b[index]!;
    return (
      event.actor === other.actor &&
      event.action === other.action &&
      event.target === other.target &&
      event.x === other.x &&
      event.y === other.y &&
      event.key === other.key &&
      event.direction === other.direction &&
      event.value === other.value
    );
  });
}
