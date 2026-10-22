import {
  MOVEMENT_DATASET_VERSION,
  episodeDuration,
  isPointerEvent,
  pointerAnchors,
  type MovementEpisode,
  type MovementEvent,
  type Point,
  type ScreenSize,
} from "./events.js";

/** Request to reproduce a learned movement, optionally retargeted to new endpoints. */
export type MovementGenerateRequest = {
  label: string;
  start?: Point;
  end?: Point;
  screen?: ScreenSize;
};

/**
 * Pluggable movement model. The in-process `TemplateMovementModel` is the
 * default; a real on-device network can implement the same contract.
 */
export interface MovementModel {
  readonly kind: string;
  train(episodes: MovementEpisode[]): MovementTrainingReport;
  labels(): string[];
  generate(request: MovementGenerateRequest): MovementEpisode;
}

export type MovementTrainingReport = {
  labels: Array<{ label: string; demonstrations: number; meanDurationMs: number }>;
  skipped: Array<{ id: string; reason: string }>;
};

/**
 * Events in a canonical frame: pointer coordinates are expressed relative to
 * the start→end segment (start = (0,0), end = (1,0)), time as progress 0..1.
 * For in-place movements (start ≈ end) coordinates are pixel offsets from start.
 */
type CanonicalEvent = { p: number; event: MovementEvent; u?: number; v?: number };

type MovementTemplate = {
  label: string;
  demonstrations: number;
  durationMs: number;
  /** Mean pointer path length of the demonstrations' start→end chord. */
  chordLength: number;
  inPlace: boolean;
  /** Medoid demonstration's events in canonical form. */
  events: CanonicalEvent[];
  screen: ScreenSize;
};

export type SerializedTemplateModel = {
  kind: "template";
  version: typeof MOVEMENT_DATASET_VERSION;
  templates: MovementTemplate[];
};

const IN_PLACE_THRESHOLD_PX = 4;
const RESAMPLE_POINTS = 32;

function toCanonical(point: Point, start: Point, end: Point, inPlace: boolean): { u: number; v: number } {
  const ox = point.x - start.x;
  const oy = point.y - start.y;
  if (inPlace) return { u: ox, v: oy };
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const len2 = dx * dx + dy * dy;
  return { u: (ox * dx + oy * dy) / len2, v: (-ox * dy + oy * dx) / len2 };
}

function fromCanonical(u: number, v: number, start: Point, end: Point, inPlace: boolean): Point {
  if (inPlace) return { x: start.x + u, y: start.y + v };
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  return { x: start.x + u * dx - v * dy, y: start.y + u * dy + v * dx };
}

/** Resample the pointer path of a canonical episode to `n` points by progress. */
function resamplePath(events: CanonicalEvent[], n = RESAMPLE_POINTS): Array<{ u: number; v: number }> {
  const path = events.filter((e) => e.u !== undefined) as Array<Required<Pick<CanonicalEvent, "p" | "u" | "v">>>;
  if (path.length === 0) return [];
  const out: Array<{ u: number; v: number }> = [];
  let j = 0;
  for (let i = 0; i < n; i += 1) {
    const p = n === 1 ? 0 : i / (n - 1);
    while (j < path.length - 2 && path[j + 1]!.p < p) j += 1;
    const a = path[j]!;
    const b = path[Math.min(j + 1, path.length - 1)]!;
    const span = b.p - a.p;
    const w = span <= 0 ? 0 : Math.min(1, Math.max(0, (p - a.p) / span));
    out.push({ u: a.u + (b.u - a.u) * w, v: a.v + (b.v - a.v) * w });
  }
  return out;
}

function pathDistance(a: Array<{ u: number; v: number }>, b: Array<{ u: number; v: number }>): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let total = 0;
  for (let i = 0; i < n; i += 1) total += Math.hypot(a[i]!.u - b[i]!.u, a[i]!.v - b[i]!.v);
  return total / n;
}

/**
 * Learns one template per label from demonstrations and generalizes by
 * similarity-transforming (rotate/scale/translate) the template onto new
 * start/end points. The template is the *medoid* demonstration — the one whose
 * canonical path is closest to all others — so discrete events (clicks, keys)
 * stay internally consistent while outlier demonstrations are rejected.
 * Duration scales with sqrt(distance ratio), a Fitts-like compromise.
 */
export class TemplateMovementModel implements MovementModel {
  readonly kind = "template";
  private templates = new Map<string, MovementTemplate>();

  train(episodes: MovementEpisode[]): MovementTrainingReport {
    const skipped: MovementTrainingReport["skipped"] = [];
    const byLabel = new Map<string, Array<{ episode: MovementEpisode; canonical: CanonicalEvent[]; chord: number; inPlace: boolean }>>();

    for (const episode of episodes) {
      const duration = episodeDuration(episode);
      if (episode.events.length === 0) {
        skipped.push({ id: episode.id, reason: "no events" });
        continue;
      }
      const anchors = pointerAnchors(episode) ?? { start: { x: 0, y: 0 }, end: { x: 0, y: 0 } };
      const chord = Math.hypot(anchors.end.x - anchors.start.x, anchors.end.y - anchors.start.y);
      const inPlace = chord < IN_PLACE_THRESHOLD_PX;
      const canonical = episode.events.map<CanonicalEvent>((event) => {
        const p = duration === 0 ? 0 : event.t / duration;
        if (!isPointerEvent(event)) return { p, event };
        return { p, event, ...toCanonical(event, anchors.start, anchors.end, inPlace) };
      });
      const group = byLabel.get(episode.label) ?? [];
      group.push({ episode, canonical, chord, inPlace });
      byLabel.set(episode.label, group);
    }

    for (const [label, group] of byLabel) {
      const inPlaceVotes = group.filter((g) => g.inPlace).length;
      const inPlace = inPlaceVotes * 2 > group.length;
      const consistent = group.filter((g) => g.inPlace === inPlace);
      for (const g of group) {
        if (g.inPlace !== inPlace) skipped.push({ id: g.episode.id, reason: "inconsistent with label's movement kind" });
      }
      const paths = consistent.map((g) => resamplePath(g.canonical));
      let medoid = 0;
      let best = Infinity;
      paths.forEach((path, i) => {
        const cost = paths.reduce((sum, other) => sum + pathDistance(path, other), 0);
        if (cost < best) {
          best = cost;
          medoid = i;
        }
      });
      const chosen = consistent[medoid]!;
      this.templates.set(label, {
        label,
        demonstrations: consistent.length,
        durationMs: consistent.reduce((s, g) => s + episodeDuration(g.episode), 0) / consistent.length,
        chordLength: consistent.reduce((s, g) => s + g.chord, 0) / consistent.length,
        inPlace,
        events: chosen.canonical,
        screen: chosen.episode.screen,
      });
    }

    return {
      labels: [...this.templates.values()].map((t) => ({
        label: t.label,
        demonstrations: t.demonstrations,
        meanDurationMs: Math.round(t.durationMs),
      })),
      skipped,
    };
  }

  labels(): string[] {
    return [...this.templates.keys()].sort();
  }

  generate(request: MovementGenerateRequest): MovementEpisode {
    const template = this.templates.get(request.label);
    if (!template) {
      throw new Error(`no learned movement for label: ${request.label}`);
    }
    const screen = request.screen ?? template.screen;
    const reference = this.referenceAnchors(template);
    const start = request.start ?? reference.start;
    const end = template.inPlace ? start : (request.end ?? reference.end);
    const chord = Math.hypot(end.x - start.x, end.y - start.y);
    if (!template.inPlace && chord < IN_PLACE_THRESHOLD_PX) {
      throw new Error(`label ${request.label} is a directed movement; start and end must differ`);
    }
    const ratio = template.inPlace || template.chordLength === 0 ? 1 : chord / template.chordLength;
    const duration = template.durationMs * Math.sqrt(ratio);

    const clamp = (point: Point): Point => ({
      x: Math.min(screen.width - 1, Math.max(0, point.x)),
      y: Math.min(screen.height - 1, Math.max(0, point.y)),
    });

    const events = template.events.map<MovementEvent>(({ p, event, u, v }) => {
      const t = Math.round(p * duration);
      if (u === undefined || v === undefined || !isPointerEvent(event)) {
        return { ...event, t } as MovementEvent;
      }
      const point = clamp(fromCanonical(u, v, start, end, template.inPlace));
      return { ...event, t, x: point.x, y: point.y } as MovementEvent;
    });

    return {
      version: MOVEMENT_DATASET_VERSION,
      id: `gen-${request.label}-${Math.round(start.x)}-${Math.round(start.y)}-${Math.round(end.x)}-${Math.round(end.y)}`,
      label: request.label,
      createdAt: new Date().toISOString(),
      source: "simulated",
      screen: { ...screen },
      events,
    };
  }

  /** Recover the medoid demonstration's original anchors (for "repeat exactly"). */
  private referenceAnchors(template: MovementTemplate): { start: Point; end: Point } {
    const pointer = template.events.find((e) => e.u !== undefined && isPointerEvent(e.event));
    const last = [...template.events].reverse().find((e) => e.u !== undefined && isPointerEvent(e.event));
    const startEvent = pointer?.event as { x: number; y: number } | undefined;
    const endEvent = last?.event as { x: number; y: number } | undefined;
    return {
      start: startEvent ? { x: startEvent.x, y: startEvent.y } : { x: 0, y: 0 },
      end: endEvent ? { x: endEvent.x, y: endEvent.y } : { x: 0, y: 0 },
    };
  }

  toJSON(): SerializedTemplateModel {
    return { kind: "template", version: MOVEMENT_DATASET_VERSION, templates: [...this.templates.values()] };
  }

  static fromJSON(data: SerializedTemplateModel): TemplateMovementModel {
    if (data.kind !== "template" || data.version !== MOVEMENT_DATASET_VERSION) {
      throw new Error("unsupported movement model artifact");
    }
    const model = new TemplateMovementModel();
    for (const template of data.templates) model.templates.set(template.label, template);
    return model;
  }
}

export type MovementEvalResult = {
  episodes: number;
  /** Mean resampled pointer error in pixels between generated and held-out paths. */
  meanErrorPx: number;
  /** meanErrorPx divided by the held-out chord length (scale-free). */
  meanNormalizedError: number;
  perEpisode: Array<{ id: string; errorPx: number; normalizedError: number }>;
};

/**
 * Generalization harness: for each held-out episode, ask the model to produce
 * the same label between that episode's own endpoints and measure path error.
 */
export function evaluateMovementModel(model: MovementModel, heldOut: MovementEpisode[]): MovementEvalResult {
  const perEpisode: MovementEvalResult["perEpisode"] = [];
  for (const episode of heldOut) {
    const anchors = pointerAnchors(episode);
    if (!anchors || !model.labels().includes(episode.label)) continue;
    const generated = model.generate({ label: episode.label, ...anchors, screen: episode.screen });
    const a = resamplePath(pixelPath(episode));
    const b = resamplePath(pixelPath(generated));
    const errorPx = pathDistance(a, b);
    const chord = Math.hypot(anchors.end.x - anchors.start.x, anchors.end.y - anchors.start.y);
    perEpisode.push({ id: episode.id, errorPx, normalizedError: chord > 0 ? errorPx / chord : 0 });
  }
  const mean = (values: number[]) => (values.length === 0 ? 0 : values.reduce((s, v) => s + v, 0) / values.length);
  return {
    episodes: perEpisode.length,
    meanErrorPx: mean(perEpisode.map((e) => e.errorPx)),
    meanNormalizedError: mean(perEpisode.map((e) => e.normalizedError)),
    perEpisode,
  };
}

function pixelPath(episode: MovementEpisode): CanonicalEvent[] {
  const duration = episodeDuration(episode);
  return episode.events.filter(isPointerEvent).map((event) => ({
    p: duration === 0 ? 0 : event.t / duration,
    event,
    u: event.x,
    v: event.y,
  }));
}
