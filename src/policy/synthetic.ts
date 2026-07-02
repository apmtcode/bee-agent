import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

// ---------------------------------------------------------------------------
// Synthetic movement event streams
//
// Deterministic generators that produce structurally-identical trajectories
// parameterized by a target object. Training on some targets and evaluating on
// a held-out target exercises the movement model's generalization (objective
// 2d) without any real OS input — which we cannot access from the cloud.
// (Math.random / Date.now are unavailable here, so everything is seeded/indexed.)
// ---------------------------------------------------------------------------

export type SynthesizeWorkflowParams = {
  target: string;
  sessionId?: string;
  startTs?: number;
  stepMs?: number;
};

/**
 * A canonical "edit-and-save" workflow: focus the editor, open the target,
 * see it rendered, type into it, then save it. Every target shares the same
 * shape; only the object slot differs — so a model that learns the shape can
 * transfer to an unseen target and slot-fill from context.
 */
export function synthesizeWorkflowReplay(params: SynthesizeWorkflowParams): ReplayManifest {
  const sessionId = params.sessionId ?? `synthetic-${params.target}`;
  const start = params.startTs ?? 0;
  const step = params.stepMs ?? 10;
  const at = (index: number) => start + step * index;

  // The target object is established in context (`select <target>`) before the
  // first action, so every action's object is recoverable from context — the
  // model can both repeat the exact sequence and transfer it to a new target.
  const span = buildTrajectorySpan({
    id: `traj-${params.target}`,
    sessionId,
    captureTier: "app",
    observations: [
      { kind: "observation", source: "os", summary: "focus editor", ts: at(1) },
      { kind: "observation", source: "os", summary: `select ${params.target}`, ts: at(2) },
      { kind: "observation", source: "os", summary: `show ${params.target}`, ts: at(4) },
    ],
    actions: [
      { kind: "action", tool: "device", summary: `open ${params.target}`, ts: at(3) },
      { kind: "action", tool: "device", summary: `type ${params.target}`, ts: at(5) },
      { kind: "action", tool: "device", summary: `save ${params.target}`, ts: at(6) },
    ],
    outcome: { status: "success", summary: `saved ${params.target}`, reward: 1 },
  });

  return buildReplayManifest({ sessionId, transcript: [], trajectories: [span] });
}

/** Build one workflow replay per target. */
export function synthesizeWorkflowDataset(targets: string[]): ReplayManifest[] {
  return targets.map((target, index) =>
    synthesizeWorkflowReplay({ target, startTs: index * 1000 }),
  );
}
