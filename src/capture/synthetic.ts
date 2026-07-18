import { buildTrajectorySpan, type CaptureTier, type TrajectorySpan } from "./trajectory.js";

/**
 * Deterministic synthetic movement-stream generator. The self-evolution engine
 * runs in the cloud with no access to the user's machine, so the movement
 * subsystem is validated against *simulated* event streams rather than real
 * mouse/keyboard/UI capture. This produces reviewable {@link TrajectorySpan}s
 * from declarative "movement programs" (named sequences of action tools), with
 * an injectable step clock so output is fully reproducible in tests/CI.
 */

export type MovementProgram = {
  /** Stable name, used to derive trajectory/session ids. */
  name: string;
  /** Ordered action tool tokens the program performs. */
  steps: string[];
  /** Optional per-step human summary; defaults to the tool token. */
  summaries?: Record<string, string>;
  /** Optional leading observation source (e.g. the app/window in focus). */
  observationSource?: string;
};

export type SyntheticMovementOptions = {
  /** Times to repeat each program (default 1). */
  repeat?: number;
  /** Milliseconds between successive events (default 100). */
  stepMillis?: number;
  /** Epoch millis for the first event (default 0 — keep tests hermetic). */
  startTs?: number;
  /** Capture tier stamped on each span (default "app"). */
  captureTier?: CaptureTier;
};

/**
 * Expand movement programs into trajectory spans. Ids are derived from the
 * program name and repetition index and event timestamps come from the
 * injectable step clock (no wall clock, no randomness), so the same inputs
 * always yield identical observation/action streams.
 */
export function generateSyntheticMovementTrajectories(
  programs: MovementProgram[],
  options: SyntheticMovementOptions = {},
): TrajectorySpan[] {
  const repeat = Math.max(1, Math.floor(options.repeat ?? 1));
  const stepMillis = Math.max(1, Math.floor(options.stepMillis ?? 100));
  const captureTier = options.captureTier ?? "app";
  const startTs = options.startTs ?? 0;

  const spans: TrajectorySpan[] = [];
  let cursor = startTs;
  for (const program of programs) {
    for (let iteration = 0; iteration < repeat; iteration += 1) {
      const id = `${program.name}-${iteration}`;
      const observations = program.observationSource
        ? [
            {
              kind: "observation" as const,
              source: program.observationSource,
              summary: `focus:${program.observationSource}`,
              ts: cursor,
            },
          ]
        : [];
      if (program.observationSource) {
        cursor += stepMillis;
      }
      const actions = program.steps.map((tool) => {
        const action = {
          kind: "action" as const,
          tool,
          summary: program.summaries?.[tool] ?? tool,
          ts: cursor,
        };
        cursor += stepMillis;
        return action;
      });
      spans.push(
        buildTrajectorySpan({
          id,
          sessionId: `session-${program.name}`,
          captureTier,
          observations,
          actions,
          outcome: { status: "success", summary: `${program.name} completed`, reward: 1 },
        }),
      );
    }
  }
  return spans;
}
