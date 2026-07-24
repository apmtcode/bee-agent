import type {
  TrajectoryAction,
  TrajectoryObservation,
  TrajectorySpan,
} from "./trajectory.js";

/**
 * Deterministic synthetic movement-event generator. Produces
 * {@link TrajectorySpan}s that stand in for real OS capture (mouse, keyboard,
 * window/UI events) so the capture -> dataset -> replay -> train/infer pipeline
 * can be exercised end to end in the cloud/CI without touching a real machine.
 *
 * "Deterministic" is load-bearing: given the same seed and options the output
 * is byte-for-byte identical, so tests and generalization evals are stable.
 */

export type SyntheticApp = {
  appId: string;
  /** Screens the app can be on; the model learns/generalizes across these. */
  screens: string[];
  /** The signature movement this app affords (the "skill" to learn). */
  skill: {
    tool: string;
    gesture: string;
    direction?: string;
    target: string;
  };
};

export const DEFAULT_SYNTHETIC_APPS: SyntheticApp[] = [
  { appId: "editor", screens: ["main.ts", "index.ts", "readme.md", "notes.txt", "config.json"], skill: { tool: "device", gesture: "type", target: "buffer" } },
  { appId: "browser", screens: ["docs", "issues", "search", "dashboard", "profile"], skill: { tool: "device", gesture: "tap", target: "link" } },
  { appId: "terminal", screens: ["build", "test", "deploy", "logs", "shell"], skill: { tool: "device", gesture: "shortcut", target: "run" } },
  { appId: "files", screens: ["downloads", "projects", "pictures", "desktop", "trash"], skill: { tool: "device", gesture: "swipe", direction: "down", target: "list" } },
];

export type SyntheticTrajectoryOptions = {
  seed?: number;
  apps?: SyntheticApp[];
  /** Trajectories to emit. */
  count?: number;
  /** Movement steps per trajectory. */
  stepsPerTrajectory?: number;
  /** Base epoch millis for the first event (kept fixed for determinism). */
  baseTs?: number;
  /** Screens (by index) to withhold per app, to build held-out eval sets. */
  holdOutScreenIndexes?: number[];
};

const DEFAULT_BASE_TS = 1_700_000_000_000;

/**
 * Generate synthetic trajectories. Screens listed in `holdOutScreenIndexes`
 * are never used, leaving them available as unseen-but-related contexts for a
 * generalization eval (same app + skill, different screen).
 */
export function synthesizeMovementTrajectories(
  options: SyntheticTrajectoryOptions = {},
): TrajectorySpan[] {
  const apps = options.apps ?? DEFAULT_SYNTHETIC_APPS;
  const count = options.count ?? 12;
  const steps = options.stepsPerTrajectory ?? 4;
  const baseTs = options.baseTs ?? DEFAULT_BASE_TS;
  const holdOut = new Set(options.holdOutScreenIndexes ?? []);
  const rng = makeRng(options.seed ?? 1);

  const spans: TrajectorySpan[] = [];

  for (let i = 0; i < count; i += 1) {
    const app = apps[Math.floor(rng() * apps.length)] ?? apps[0];
    if (!app) {
      break;
    }
    const availableScreens = app.screens.filter((_, index) => !holdOut.has(index));
    const observations: TrajectoryObservation[] = [];
    const actions: TrajectoryAction[] = [];
    let ts = baseTs + i * 1_000_000;

    for (let step = 0; step < steps; step += 1) {
      const screen = availableScreens[Math.floor(rng() * availableScreens.length)] ?? app.screens[0] ?? "screen";
      observations.push({
        kind: "observation",
        source: "os",
        summary: `focused ${app.appId} on ${screen}`,
        ts,
        metadata: { event: "focus-changed", appId: app.appId, screenTitle: screen },
      });
      ts += 500;
      actions.push({
        kind: "action",
        tool: app.skill.tool,
        summary: `${app.skill.gesture} ${app.skill.target} on ${screen}`,
        ts,
        metadata: {
          gesture: app.skill.gesture,
          target: app.skill.target,
          ...(app.skill.direction ? { direction: app.skill.direction } : {}),
        },
      });
      ts += 500;
    }

    spans.push({
      id: `synthetic-${i}`,
      sessionId: `synthetic-session-${i % 3}`,
      createdAt: new Date(baseTs + i).toISOString(),
      captureTier: "app",
      observations,
      actions,
      outcome: { status: "success", summary: `used ${app.appId}` },
    });
  }

  return spans;
}

/** Deterministic LCG — avoids Math.random so runs are reproducible. */
function makeRng(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}
