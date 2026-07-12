import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Deterministic synthetic movement generator.
 *
 * bee-agent has no access to a real machine in the cloud, so we validate the
 * capture -> dataset -> train -> replay pipeline against simulated event streams
 * instead. This produces reproducible trajectory spans from a seed (a small LCG,
 * no `Math.random`) so tests are stable and generalization can be measured on
 * held-out-but-related trajectories drawn from the same grammar.
 */

export type SyntheticMovementOptions = {
  seed: number;
  /** Number of trajectories to emit. */
  count: number;
  /** Movements per trajectory (min inclusive, max inclusive). */
  lengthRange?: [number, number];
  /** Interaction "app" the movements belong to. */
  app?: string;
};

type Step = { tool: string; summary: string };

/**
 * A tiny grammar of UI movements. Sequences follow a plausible order
 * (focus -> navigate -> type -> click -> submit) so a trained model has real
 * structure to learn and generalize over rather than noise.
 */
const MOVEMENT_GRAMMAR: Step[][] = [
  [
    { tool: "window", summary: "focus editor" },
    { tool: "keyboard", summary: "type command" },
    { tool: "keyboard", summary: "press enter" },
    { tool: "window", summary: "observe output" },
  ],
  [
    { tool: "mouse", summary: "move cursor" },
    { tool: "mouse", summary: "click button" },
    { tool: "window", summary: "wait dialog" },
    { tool: "mouse", summary: "click confirm" },
  ],
  [
    { tool: "browser", summary: "open page" },
    { tool: "keyboard", summary: "type query" },
    { tool: "mouse", summary: "click result" },
    { tool: "browser", summary: "read content" },
  ],
];

function lcg(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export function generateSyntheticTrajectories(options: SyntheticMovementOptions): TrajectorySpan[] {
  const next = lcg(options.seed);
  const [minLen, maxLen] = options.lengthRange ?? [3, 6];
  const app = options.app ?? "sim-app";
  const trajectories: TrajectorySpan[] = [];

  for (let index = 0; index < options.count; index += 1) {
    const grammar = MOVEMENT_GRAMMAR[Math.floor(next() * MOVEMENT_GRAMMAR.length) % MOVEMENT_GRAMMAR.length];
    const length = minLen + Math.floor(next() * (maxLen - minLen + 1));
    let ts = 1000;
    const actions = Array.from({ length }, (_unused, step) => {
      const template = grammar[step % grammar.length];
      ts += 10 + Math.floor(next() * 40);
      return {
        kind: "action" as const,
        tool: template.tool,
        summary: template.summary,
        ts,
        metadata: { app, step },
      };
    });

    trajectories.push(
      buildTrajectorySpan({
        id: `${app}-traj-${index}`,
        sessionId: `${app}-session`,
        captureTier: "app",
        observations: [
          { kind: "observation", source: app, summary: "session started", ts: 1000, metadata: { app } },
        ],
        actions,
        outcome: { status: "success", summary: "sequence completed", reward: 1 },
      }),
    );
  }

  return trajectories;
}
