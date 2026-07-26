import type { DeviceCaptureInput, DevicePlatform } from "./device-adapter.js";
import type { OsObservationInput } from "./os-observer.js";
import type { CaptureTier, TrajectorySpan } from "./trajectory.js";

/**
 * Synthetic movement event-stream generator.
 *
 * bee-agent runs in Anthropic's cloud with no access to a real machine, so the
 * local-movement learning subsystem cannot be validated against genuine OS
 * input. This module produces deterministic, seeded streams of device gestures
 * and OS events shaped exactly like the inputs the real capture adapters
 * consume ({@link DeviceCaptureInput} / {@link OsObservationInput}), so the
 * capture -> dataset -> replay -> train round-trip can be exercised and tested
 * end-to-end without touching a physical device.
 *
 * Determinism is a hard requirement: given the same (scenario, seed, startTs)
 * the generator emits byte-identical streams. That is what lets a "family" of
 * streams share a task structure while varying targets/values by seed — the
 * substrate a generalization eval needs ("perform new but related movements").
 */

/** A pure function returning the next pseudo-random float in [0, 1). */
export type Rng = () => number;

/**
 * mulberry32 — a tiny, fast, fully deterministic 32-bit PRNG. Seeded, so it
 * never touches {@link Math.random}; identical seeds yield identical sequences.
 */
export function createRng(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SyntheticStreamEvent =
  | { channel: "device"; input: DeviceCaptureInput }
  | { channel: "os"; input: OsObservationInput };

export type SyntheticStream = {
  version: 1;
  scenario: string;
  seed: number;
  sessionId: string;
  deviceId: string;
  platform: DevicePlatform;
  captureTier: CaptureTier;
  startTs: number;
  endTs: number;
  events: SyntheticStreamEvent[];
};

export type SyntheticStreamOptions = {
  scenario: string;
  seed: number;
  /** Fixed default epoch keeps generation deterministic (no wall-clock reads). */
  startTs?: number;
  sessionId?: string;
  deviceId?: string;
  platform?: DevicePlatform;
  captureTier?: CaptureTier;
  visibleIndicator?: boolean;
  minDelayMs?: number;
  maxDelayMs?: number;
};

/** Deterministic default so `startTs`-less generation stays reproducible. */
export const SYNTHETIC_STREAM_DEFAULT_START_TS = 1_700_000_000_000;

const DEFAULT_MIN_DELAY_MS = 120;
const DEFAULT_MAX_DELAY_MS = 900;

const APP_POOLS = {
  editor: ["vscode", "sublime-text", "neovim", "zed"],
  browser: ["chrome", "firefox", "safari", "arc"],
  terminal: ["iterm", "terminal", "warp"],
  chat: ["slack", "discord", "messages"],
} as const;

const FILE_POOL = [
  "src/index.ts",
  "src/capture/recorder.ts",
  "README.md",
  "notes/plan.md",
  "config/settings.yaml",
];

const QUERY_POOL = [
  "seeded prng typescript",
  "vitest fake timers",
  "replay manifest schema",
  "mlx lora fine-tuning",
  "deterministic event stream",
];

const SNIPPET_POOL = [
  "const rng = createRng(seed)",
  "export function replay() {}",
  "// TODO: wire up adapter",
  "await store.add(trajectory)",
];

/** Internal builder that turns semantic steps into adapter-shaped events. */
class StreamBuilder {
  private clock: number;
  private readonly events: SyntheticStreamEvent[] = [];
  private currentApp: string;

  constructor(
    private readonly rng: Rng,
    private readonly ctx: {
      sessionId: string;
      deviceId: string;
      platform: DevicePlatform;
      captureTier: CaptureTier;
      visibleIndicator: boolean;
      minDelayMs: number;
      maxDelayMs: number;
      startTs: number;
      initialApp: string;
    },
  ) {
    this.clock = ctx.startTs;
    this.currentApp = ctx.initialApp;
  }

  private advance(): number {
    const span = Math.max(0, this.ctx.maxDelayMs - this.ctx.minDelayMs);
    const delay = this.ctx.minDelayMs + Math.floor(this.rng() * (span + 1));
    this.clock += delay;
    return this.clock;
  }

  focus(appId: string, windowTitle: string): this {
    this.currentApp = appId;
    const ts = this.advance();
    this.events.push({
      channel: "os",
      input: {
        sessionId: this.ctx.sessionId,
        appId,
        visibleIndicator: this.ctx.visibleIndicator,
        captureTier: this.ctx.captureTier,
        ts,
        event: "focus-changed",
        windowTitle,
      },
    });
    return this;
  }

  openFile(filePath: string): this {
    const ts = this.advance();
    this.events.push({
      channel: "os",
      input: {
        sessionId: this.ctx.sessionId,
        appId: this.currentApp,
        visibleIndicator: this.ctx.visibleIndicator,
        captureTier: this.ctx.captureTier,
        ts,
        event: "file-opened",
        filePath,
      },
    });
    return this;
  }

  runCommand(commandSummary: string): this {
    const ts = this.advance();
    this.events.push({
      channel: "os",
      input: {
        sessionId: this.ctx.sessionId,
        appId: this.currentApp,
        visibleIndicator: this.ctx.visibleIndicator,
        captureTier: this.ctx.captureTier,
        ts,
        event: "command-ran",
        commandSummary,
      },
    });
    return this;
  }

  private gesture(
    gesture: NonNullable<DeviceCaptureInput["gesture"]>,
    screenTitle: string,
  ): this {
    const ts = this.advance();
    this.events.push({
      channel: "device",
      input: {
        sessionId: this.ctx.sessionId,
        deviceId: this.ctx.deviceId,
        platform: this.ctx.platform,
        appId: this.currentApp,
        appName: appDisplayName(this.currentApp),
        screenTitle,
        captureTier: this.ctx.captureTier,
        visibleIndicator: this.ctx.visibleIndicator,
        ts,
        gesture: { ...gesture, ts },
      },
    });
    return this;
  }

  tap(target: string, screenTitle: string): this {
    return this.gesture({ kind: "tap", target, ts: 0 }, screenTitle);
  }

  type(target: string, valueSummary: string, screenTitle: string): this {
    return this.gesture({ kind: "type", target, valueSummary, ts: 0 }, screenTitle);
  }

  scroll(direction: "up" | "down" | "left" | "right", screenTitle: string): this {
    return this.gesture({ kind: "scroll", direction, ts: 0 }, screenTitle);
  }

  shortcut(target: string, screenTitle: string): this {
    return this.gesture({ kind: "shortcut", target, ts: 0 }, screenTitle);
  }

  /** Attach an outcome to the most recent event (device or os). */
  finish(outcome: NonNullable<TrajectorySpan["outcome"]>): this {
    const last = this.events[this.events.length - 1];
    if (last) {
      last.input.outcome = outcome;
    }
    return this;
  }

  build(): { events: SyntheticStreamEvent[]; endTs: number } {
    return { events: this.events, endTs: this.clock };
  }
}

function appDisplayName(appId: string): string {
  return appId
    .split(/[-_]/)
    .map((part) => (part.length > 0 ? part[0]!.toUpperCase() + part.slice(1) : part))
    .join(" ");
}

function pick<T>(rng: Rng, pool: readonly T[]): T {
  return pool[Math.floor(rng() * pool.length)]!;
}

type ScenarioBuilder = (builder: StreamBuilder, rng: Rng) => void;

/**
 * Scenario registry. Each scenario emits a coherent task made of the same
 * ordered *shape* of actions on every seed (so {@link streamActionSignature}
 * is seed-invariant), while its concrete targets/values vary by seed — the
 * "related but new movements" property the generalization objective needs.
 */
const SCENARIOS: Record<string, ScenarioBuilder> = {
  "edit-file": (builder, rng) => {
    const editor = pick(rng, APP_POOLS.editor);
    const file = pick(rng, FILE_POOL);
    const snippet = pick(rng, SNIPPET_POOL);
    builder
      .focus(editor, appDisplayName(editor))
      .openFile(file)
      .tap(file, file)
      .type(file, snippet, file)
      .shortcut("save", file)
      .finish({ status: "success", summary: `edited ${file}`, reward: 1 });
  },
  "web-search": (builder, rng) => {
    const browser = pick(rng, APP_POOLS.browser);
    const query = pick(rng, QUERY_POOL);
    builder
      .focus(browser, appDisplayName(browser))
      .tap("address-bar", "New Tab")
      .type("address-bar", query, "New Tab")
      .tap("result-1", `results: ${query}`)
      .scroll("down", `results: ${query}`)
      .finish({ status: "success", summary: `searched ${query}`, reward: 1 });
  },
  "switch-and-copy": (builder, rng) => {
    const source = pick(rng, APP_POOLS.browser);
    const target = pick(rng, APP_POOLS.editor);
    const snippet = pick(rng, SNIPPET_POOL);
    builder
      .focus(source, appDisplayName(source))
      .tap("selection", `selecting in ${appDisplayName(source)}`)
      .shortcut("copy", appDisplayName(source))
      .focus(target, appDisplayName(target))
      .type("cursor", snippet, appDisplayName(target))
      .shortcut("paste", appDisplayName(target))
      .finish({ status: "success", summary: `copied into ${target}`, reward: 1 });
  },
  "run-command": (builder, rng) => {
    const terminal = pick(rng, APP_POOLS.terminal);
    const command = pick(rng, ["npm test", "git status", "npm run build", "vitest run"]);
    builder
      .focus(terminal, appDisplayName(terminal))
      .tap("prompt", appDisplayName(terminal))
      .type("prompt", command, appDisplayName(terminal))
      .runCommand(command)
      .finish({ status: "success", summary: `ran ${command}`, reward: 1 });
  },
};

/** Ordered list of registered scenario ids. */
export function listSyntheticScenarios(): string[] {
  return Object.keys(SCENARIOS);
}

/**
 * Generate one deterministic synthetic movement stream. The returned events
 * are ready to feed straight into {@link DeviceCaptureAdapter.record} /
 * {@link OsCaptureObserver.observe} by dispatching on `event.channel`.
 */
export function generateSyntheticStream(options: SyntheticStreamOptions): SyntheticStream {
  const scenario = SCENARIOS[options.scenario];
  if (!scenario) {
    throw new Error(
      `unknown synthetic scenario "${options.scenario}" (known: ${listSyntheticScenarios().join(", ")})`,
    );
  }

  const startTs = options.startTs ?? SYNTHETIC_STREAM_DEFAULT_START_TS;
  const platform = options.platform ?? "macos";
  const captureTier = options.captureTier ?? "app";
  const sessionId = options.sessionId ?? `synthetic-${options.scenario}-${options.seed}`;
  const deviceId = options.deviceId ?? `synthetic-device-${options.seed}`;
  const rng = createRng(options.seed);

  const builder = new StreamBuilder(rng, {
    sessionId,
    deviceId,
    platform,
    captureTier,
    visibleIndicator: options.visibleIndicator ?? true,
    minDelayMs: options.minDelayMs ?? DEFAULT_MIN_DELAY_MS,
    maxDelayMs: options.maxDelayMs ?? DEFAULT_MAX_DELAY_MS,
    startTs,
    initialApp: "desktop",
  });

  scenario(builder, rng);
  const { events, endTs } = builder.build();

  return {
    version: 1,
    scenario: options.scenario,
    seed: options.seed,
    sessionId,
    deviceId,
    platform,
    captureTier,
    startTs,
    endTs,
    events,
  };
}

/**
 * Generate a family of related streams for the same scenario across a
 * contiguous seed range — the natural train/held-out split for a
 * generalization eval. Same scenario => same action signature, distinct
 * targets/values per seed.
 */
export function generateSyntheticStreamFamily(options: {
  scenario: string;
  count: number;
  baseSeed?: number;
  startTs?: number;
  captureTier?: CaptureTier;
}): SyntheticStream[] {
  const baseSeed = options.baseSeed ?? 1;
  const streams: SyntheticStream[] = [];
  for (let i = 0; i < options.count; i += 1) {
    streams.push(
      generateSyntheticStream({
        scenario: options.scenario,
        seed: baseSeed + i,
        startTs: options.startTs,
        captureTier: options.captureTier,
      }),
    );
  }
  return streams;
}

/**
 * Compact, seed-invariant fingerprint of a stream's *movement shape*: the
 * ordered list of `channel:kind` tokens (e.g. `os:focus-changed`,
 * `device:tap`). Two streams from the same scenario share a signature
 * regardless of seed; a model that reproduces the signature on a held-out seed
 * has generalized the task structure. The building block for the eval harness.
 */
export function streamActionSignature(stream: SyntheticStream): string[] {
  return stream.events.map((event) => {
    if (event.channel === "os") {
      return `os:${event.input.event}`;
    }
    return `device:${event.input.gesture?.kind ?? "none"}`;
  });
}
