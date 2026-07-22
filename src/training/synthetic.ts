import type { ReviewedExportManifest, ExportedReplayManifest } from "./export-manifest.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud and cannot capture real OS input, so this
 * produces reproducible `ReviewedExportManifest`-shaped data (observation →
 * action timelines) to validate the capture → dataset → train → infer → evaluate
 * loop offline. It is seeded and uses a tiny LCG — no `Math.random`, so a given
 * seed always yields the same streams.
 */

export type SynthesizeMovementStreamsOptions = {
  seed?: number;
  trajectoryCount?: number;
  /** Number of observation→action steps per trajectory. */
  stepsPerTrajectory?: number;
  /** Optional label mixed into event text so related runs share vocabulary. */
  scenario?: string;
};

const TOOLS = ["mouse.click", "mouse.move", "keyboard.type", "window.focus", "scroll.wheel"] as const;
const TARGETS = ["toolbar", "canvas", "sidebar", "menu", "textfield", "button", "tab"] as const;

class Lcg {
  constructor(private state: number) {
    // Keep state in a stable positive 32-bit range regardless of the seed.
    this.state = (Math.abs(Math.trunc(state)) % 2147483647) || 1;
  }
  next(): number {
    // Park–Miller minimal standard generator; fully deterministic.
    this.state = (this.state * 16807) % 2147483647;
    return this.state / 2147483647;
  }
  pick<T>(items: readonly T[]): T {
    return items[Math.floor(this.next() * items.length) % items.length]!;
  }
  int(maxExclusive: number): number {
    return Math.floor(this.next() * maxExclusive);
  }
}

/**
 * Build a reviewed export manifest containing purely synthetic replay
 * timelines. The output is safe to feed straight into
 * {@link extractMovementDataset}.
 */
export function synthesizeMovementManifest(
  options: SynthesizeMovementStreamsOptions = {},
): ReviewedExportManifest {
  const rng = new Lcg(options.seed ?? 1);
  const trajectoryCount = Math.max(1, options.trajectoryCount ?? 3);
  const stepsPerTrajectory = Math.max(1, options.stepsPerTrajectory ?? 4);
  const scenario = options.scenario ?? "workspace";

  const replays: ExportedReplayManifest[] = [];

  for (let t = 0; t < trajectoryCount; t += 1) {
    const trajectoryId = `syn-traj-${scenario}-${t}`;
    const sessionId = `syn-session-${scenario}-${t}`;
    const events: ExportedReplayManifest["events"] = [];
    let ts = 1_000 + t * 10_000;

    events.push({
      kind: "transcript",
      ts,
      messageId: `${trajectoryId}-goal`,
      role: "user",
      content: `complete the ${scenario} flow`,
    });

    for (let step = 0; step < stepsPerTrajectory; step += 1) {
      const target = rng.pick(TARGETS);
      const tool = rng.pick(TOOLS);
      const x = rng.int(1920);
      const y = rng.int(1080);
      ts += 50 + rng.int(120);
      events.push({
        kind: "observation",
        ts,
        trajectoryId,
        source: "ui.snapshot",
        summary: `${scenario} ${target} visible at step ${step}`,
      });
      ts += 20 + rng.int(60);
      events.push({
        kind: "action",
        ts,
        trajectoryId,
        tool,
        summary: `${tool} on ${target} at (${x},${y})`,
      });
    }

    replays.push({ sessionId, trajectoryIds: [trajectoryId], eventCount: events.length, events });
  }

  const createdAt = "2026-01-01T00:00:00.000Z";
  return {
    version: 1,
    createdAt,
    reviewedBy: "synthetic-generator",
    purpose: `synthetic ${scenario} movement streams`,
    targetPlatform: "apple-silicon",
    modes: ["sft"],
    rawCaptureIncluded: false,
    promotedSkills: [],
    executableSkills: [],
    executableSkillRuns: [],
    memories: [],
    trajectories: replays.map((replay, index) => ({
      id: replay.trajectoryIds[0]!,
      sessionId: replay.sessionId,
      createdAt,
      captureTier: "app",
      observationCount: replay.events.filter((event) => event.kind === "observation").length,
      actionCount: replay.events.filter((event) => event.kind === "action").length,
      outcomeStatus: "success" as const,
      reward: 1,
      reviewedAt: createdAt,
      reviewedBy: "synthetic-generator",
      // index kept for stable ordering; unused downstream.
      ...(index >= 0 ? {} : {}),
    })),
    replays,
  };
}
