import type { MovementEpisode, MovementEvent } from "./events.js";

/**
 * Splits one long recording into labelled sub-gestures so a single capture
 * session yields many training demonstrations instead of one opaque episode.
 */
export type MovementSegmentKind = "drag" | "click" | "move-to" | "move" | "scroll" | "typing" | "chord";

export type SegmentEpisodeOptions = {
  /** A pause longer than this (ms) between events ends the current segment. */
  idleGapMs?: number;
  /** Press→release displacement (px) above which a press is a drag, not a click. */
  dragThresholdPx?: number;
  /** Pointer-move runs whose start→end distance is below this (px) are jitter and dropped. */
  minMoveDistancePx?: number;
  /** Prepend the most recent `window.focus` (at t=0) so segments keep their app context. */
  includeFocus?: boolean;
  /** Only emit these kinds. */
  kinds?: ReadonlyArray<MovementSegmentKind>;
  /** Label for each segment; defaults to `<episode.label>/<kind>`. */
  label?: (kind: MovementSegmentKind, episode: MovementEpisode) => string;
};

export type SegmentEpisodeResult = {
  segments: MovementEpisode[];
  counts: Partial<Record<MovementSegmentKind, number>>;
  /** Events not assigned to any emitted segment (jitter, unreleased presses, filtered kinds). */
  droppedEvents: number;
};

type FocusEvent = Extract<MovementEvent, { type: "window.focus" }>;
type Draft = { kind: MovementSegmentKind; events: MovementEvent[]; focus?: FocusEvent };

const isKey = (e: MovementEvent) => e.type === "key.down" || e.type === "key.up";

export function segmentEpisode(episode: MovementEpisode, options: SegmentEpisodeOptions = {}): SegmentEpisodeResult {
  const idleGap = options.idleGapMs ?? 400;
  const dragThreshold = options.dragThresholdPx ?? 8;
  const minMove = options.minMoveDistancePx ?? 4;
  const includeFocus = options.includeFocus ?? true;
  const allowKinds = options.kinds ? new Set(options.kinds) : undefined;
  const labelFor = options.label ?? ((kind, ep) => `${ep.label}/${kind}`);

  const events = episode.events;
  const drafts: Draft[] = [];
  let focus: FocusEvent | undefined;
  let dropped = 0;
  let i = 0;

  const gapOk = (prev: MovementEvent, next: MovementEvent) => next.t - prev.t <= idleGap;

  while (i < events.length) {
    const event = events[i]!;

    if (event.type === "window.focus") {
      focus = event;
      i += 1;
      continue;
    }

    if (event.type === "pointer.down") {
      // Press → (moves) → matching release. Unreleased presses are dropped.
      const run: MovementEvent[] = [event];
      let j = i + 1;
      let released = false;
      while (j < events.length) {
        const next = events[j]!;
        if (next.type === "window.focus") break;
        run.push(next);
        j += 1;
        if (next.type === "pointer.up" && next.button === event.button) {
          released = true;
          break;
        }
      }
      if (!released) {
        dropped += run.length;
      } else {
        const up = run[run.length - 1] as Extract<MovementEvent, { type: "pointer.up" }>;
        const distance = Math.hypot(up.x - event.x, up.y - event.y);
        drafts.push({ kind: distance > dragThreshold ? "drag" : "click", events: run, focus });
      }
      i = j;
      continue;
    }

    if (event.type === "pointer.move") {
      const run: MovementEvent[] = [event];
      let j = i + 1;
      while (j < events.length && events[j]!.type === "pointer.move" && gapOk(events[j - 1]!, events[j]!)) {
        run.push(events[j]!);
        j += 1;
      }
      const last = run[run.length - 1] as Extract<MovementEvent, { type: "pointer.move" }>;
      const first = event;
      if (Math.hypot(last.x - first.x, last.y - first.y) < minMove) {
        dropped += run.length;
      } else {
        const next = events[j];
        const targeted = next?.type === "pointer.down" && gapOk(last, next);
        drafts.push({ kind: targeted ? "move-to" : "move", events: run, focus });
      }
      i = j;
      continue;
    }

    if (event.type === "scroll") {
      const run: MovementEvent[] = [event];
      let j = i + 1;
      while (j < events.length && events[j]!.type === "scroll" && gapOk(events[j - 1]!, events[j]!)) {
        run.push(events[j]!);
        j += 1;
      }
      drafts.push({ kind: "scroll", events: run, focus });
      i = j;
      continue;
    }

    // Key events: a burst continues across short gaps, and always while a key
    // is held so modifier+pointer chords (shift-click) stay one balanced segment.
    const run: MovementEvent[] = [];
    const held = new Set<string>();
    let sawPointer = false;
    let j = i;
    while (j < events.length) {
      const next = events[j]!;
      if (next.type === "window.focus") break;
      if (run.length > 0) {
        const prev = run[run.length - 1]!;
        if (held.size === 0 && (!isKey(next) || !gapOk(prev, next))) break;
      }
      run.push(next);
      if (next.type === "key.down") held.add(next.key);
      else if (next.type === "key.up") held.delete(next.key);
      else sawPointer = true;
      j += 1;
    }
    drafts.push({ kind: sawPointer ? "chord" : "typing", events: run, focus });
    i = j;
  }

  const segments: MovementEpisode[] = [];
  const counts: Partial<Record<MovementSegmentKind, number>> = {};
  for (const draft of drafts) {
    if (allowKinds && !allowKinds.has(draft.kind)) {
      dropped += draft.events.length;
      continue;
    }
    const t0 = draft.events[0]!.t;
    const rebased = draft.events.map((e) => ({ ...e, t: e.t - t0 }) as MovementEvent);
    const withFocus = includeFocus && draft.focus ? [{ ...draft.focus, t: 0 }, ...rebased] : rebased;
    segments.push({
      version: episode.version,
      id: `${episode.id}#${segments.length}`,
      label: labelFor(draft.kind, episode),
      createdAt: episode.createdAt,
      source: episode.source,
      screen: { ...episode.screen },
      events: withFocus,
    });
    counts[draft.kind] = (counts[draft.kind] ?? 0) + 1;
  }

  return { segments, counts, droppedEvents: dropped };
}
