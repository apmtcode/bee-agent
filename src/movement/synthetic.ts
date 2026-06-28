import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectoryAction, TrajectoryObservation, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Deterministic synthetic event-stream generator.
 *
 * The real capture pipeline records OS/device/browser input, which is
 * unavailable in the cloud. This generator fabricates reproducible trajectory
 * spans — same seed yields the same stream — so the dataset/backend/eval code
 * can be exercised end-to-end without any real input. It models a small grammar
 * of "open app -> act on a target" movements drawn from per-app target/gesture
 * vocabularies.
 */

/**
 * A single learnable step: within an app, a given screen deterministically maps
 * to one canonical (target, gesture). This makes `context -> action` a genuine
 * function so a model can be measured on whether it recovers it.
 */
export type SyntheticStep = {
  screen: string;
  target: string;
  gesture: string;
};

export type SyntheticAppProfile = {
  appId: string;
  appName: string;
  steps: SyntheticStep[];
};

export type SyntheticStreamOptions = {
  seed: number;
  trajectoryCount: number;
  actionsPerTrajectory?: number;
  apps?: SyntheticAppProfile[];
  /** Mark generated trajectories as approved so they pass the dataset gate. */
  approved?: boolean;
  /** Base timestamp; each event advances deterministically from it. */
  startTs?: number;
};

export const DEFAULT_SYNTHETIC_APPS: SyntheticAppProfile[] = [
  {
    appId: "com.example.mail",
    appName: "Mail",
    steps: [
      { screen: "Inbox", target: "Archive", gesture: "swipe" },
      { screen: "Compose", target: "Send", gesture: "tap" },
      { screen: "Subject", target: "Subject field", gesture: "type" },
      { screen: "Thread", target: "Reply", gesture: "tap" },
    ],
  },
  {
    appId: "com.example.files",
    appName: "Files",
    steps: [
      { screen: "Browser", target: "New Folder", gesture: "tap" },
      { screen: "Preview", target: "Rename", gesture: "tap" },
      { screen: "Search", target: "Search field", gesture: "type" },
      { screen: "Selection", target: "Move", gesture: "swipe" },
    ],
  },
];

/** Small deterministic PRNG (mulberry32) — no Math.random, fully reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(rng: () => number, items: T[]): T {
  return items[Math.floor(rng() * items.length) % items.length];
}

function summarizeGesture(gesture: string, target: string): string {
  switch (gesture) {
    case "tap":
      return `tapped ${target}`;
    case "type":
      return `typed into ${target}`;
    case "scroll":
      return `scrolled ${target}`;
    case "swipe":
      return `swiped ${target}`;
    default:
      return `acted on ${target}`;
  }
}

/**
 * Generate a deterministic list of approved trajectory spans suitable for
 * {@link buildMovementDataset}.
 */
export function generateSyntheticTrajectories(options: SyntheticStreamOptions): TrajectorySpan[] {
  const rng = mulberry32(options.seed);
  const apps = options.apps ?? DEFAULT_SYNTHETIC_APPS;
  const actionsPerTrajectory = options.actionsPerTrajectory ?? 3;
  const approved = options.approved ?? true;
  let ts = options.startTs ?? 1_700_000_000_000;

  const trajectories: TrajectorySpan[] = [];
  for (let t = 0; t < options.trajectoryCount; t += 1) {
    const app = pick(rng, apps);
    const observations: TrajectoryObservation[] = [];
    const actions: TrajectoryAction[] = [];

    for (let a = 0; a < actionsPerTrajectory; a += 1) {
      const step = pick(rng, app.steps);
      ts += 1000;
      observations.push({
        kind: "observation",
        source: "device",
        summary: `${app.appName} on ${step.screen}`,
        ts,
        metadata: { appName: app.appName, screenTitle: step.screen, platform: "macos" },
      });
      ts += 500;
      actions.push({
        kind: "action",
        tool: "device",
        summary: summarizeGesture(step.gesture, step.target),
        ts,
        metadata: { gesture: step.gesture, target: step.target },
      });
    }

    const id = `syn-${options.seed}-${t}`;
    const span = buildTrajectorySpan({
      id,
      sessionId: `syn-session-${options.seed}`,
      captureTier: "full",
      observations,
      actions,
      outcome: { status: "success", summary: `completed ${app.appName} flow`, reward: 1 },
    });
    // Override the wall-clock createdAt so the stream is fully reproducible.
    span.createdAt = new Date(observations[0]?.ts ?? ts).toISOString();
    if (approved) {
      span.review = {
        status: "approved",
        reviewedAt: new Date(ts).toISOString(),
        reviewedBy: "synthetic-generator",
      };
    }
    trajectories.push(span);
  }

  return trajectories;
}
