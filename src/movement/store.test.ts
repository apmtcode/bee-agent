import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { FileCaptureConsentStore } from "../capture/consent-store.js";
import { defaultCapturePolicy, type CapturePolicy } from "../capture/trajectory.js";
import type { MovementEpisode } from "./events.js";
import { TemplateMovementModel } from "./model.js";
import { SimulatedInputEventSource, recordEpisode, synthesizeDrag } from "./source.js";
import { FileMovementDatasetStore, MovementConsentError, movementLabelFileStem } from "./store.js";

const NOW = new Date("2026-10-27T12:00:00.000Z");

async function drag(id: string, label: string, createdAt: Date, seed = 1): Promise<MovementEpisode> {
  const source = new SimulatedInputEventSource(
    synthesizeDrag({ from: { x: 100, y: 100 }, to: { x: 400 + seed * 10, y: 220 }, seed, durationMs: 400 }),
  );
  return await recordEpisode(source, { id, label, now: () => createdAt });
}

describe("FileMovementDatasetStore", () => {
  let root: string;
  let consent: FileCaptureConsentStore;
  let policy: CapturePolicy;
  let store: FileMovementDatasetStore;

  beforeEach(async () => {
    root = await fs.promises.mkdtemp(path.join(os.tmpdir(), "movement-store-"));
    consent = new FileCaptureConsentStore(path.join(root, "consent.json"));
    policy = { ...defaultCapturePolicy(), retainRawDays: 7 };
    store = new FileMovementDatasetStore({
      rootDir: path.join(root, "movement"),
      consent,
      policy: () => policy,
      now: () => NOW,
    });
  });

  afterEach(async () => {
    await fs.promises.rm(root, { recursive: true, force: true });
  });

  it("refuses to record without an active grant at a recording tier", async () => {
    const episode = await drag("e1", "drag-file", NOW);
    await expect(store.append([episode])).rejects.toBeInstanceOf(MovementConsentError);
    await consent.create({ tier: "operator", purpose: "operator only" });
    await expect(store.append([episode])).rejects.toThrow(/tier app or full/);
    const expired = await consent.create({ tier: "full", purpose: "old", expiresAt: "2026-01-01T00:00:00.000Z" });
    expect(expired.tier).toBe("full");
    await expect(store.append([episode])).rejects.toBeInstanceOf(MovementConsentError);
    expect(fs.existsSync(path.join(root, "movement"))).toBe(false);
  });

  it("appends across calls into one JSONL per label, with a single header", async () => {
    const grant = await consent.create({ tier: "app", purpose: "learn drags" });
    const first = await store.append([await drag("e1", "drag-file", NOW, 1)]);
    expect(first.grantId).toBe(grant.id);
    await store.append([await drag("e2", "drag-file", NOW, 2), await drag("c1", "Close Tab", NOW, 3)]);

    const raw = await fs.promises.readFile(store.datasetPath("drag-file"), "utf8");
    expect(raw.match(/"movement-dataset"/g)).toHaveLength(1);
    expect((await store.readEpisodes("drag-file")).map((e) => e.id)).toEqual(["e1", "e2"]);

    const summaries = await store.listDatasets();
    expect(summaries.map((s) => [s.label, s.episodes])).toEqual([
      ["Close Tab", 1],
      ["drag-file", 2],
    ]);
    expect(summaries[0]!.newestAt).toBe(NOW.toISOString());
  });

  it("rejects invalid episodes individually without blocking valid ones", async () => {
    await consent.create({ tier: "full", purpose: "x" });
    const good = await drag("ok", "drag-file", NOW);
    const bad = { ...good, id: "bad", events: [{ type: "teleport", t: 0 }] } as unknown as MovementEpisode;
    const result = await store.append([good, bad]);
    expect(result.appended.map((a) => a.id)).toEqual(["ok"]);
    expect(result.rejected[0]!.id).toBe("bad");
    expect(await store.readEpisodes("drag-file")).toHaveLength(1);
  });

  it("serializes concurrent appends so no lines interleave or duplicate headers", async () => {
    await consent.create({ tier: "app", purpose: "x" });
    const episodes = await Promise.all([1, 2, 3, 4, 5].map((n) => drag(`e${n}`, "drag-file", NOW, n)));
    await Promise.all(episodes.map((episode) => store.append([episode])));
    const raw = await fs.promises.readFile(store.datasetPath("drag-file"), "utf8");
    expect(raw.match(/"movement-dataset"/g)).toHaveLength(1);
    expect((await store.readEpisodes("drag-file")).map((e) => e.id).sort()).toEqual(["e1", "e2", "e3", "e4", "e5"]);
  });

  it("gates training reads on allowTrainingExport, then trains a model that round-trips", async () => {
    await consent.create({ tier: "app", purpose: "x" });
    await store.append(await Promise.all([1, 2, 3].map((n) => drag(`e${n}`, "drag-file", NOW, n))));
    await expect(store.loadTrainingEpisodes()).rejects.toThrow(/allowTrainingExport/);

    policy = { ...policy, allowTrainingExport: true };
    const episodes = await store.loadTrainingEpisodes();
    expect(episodes).toHaveLength(3);

    const model = new TemplateMovementModel();
    model.train(episodes);
    const file = await store.saveModel("drag-v1", model);
    expect(file.endsWith(path.join("models", "drag-v1.json"))).toBe(true);
    const loaded = await store.loadModel("drag-v1");
    expect(loaded?.labels()).toEqual(["drag-file"]);
    expect(loaded?.generate({ label: "drag-file", start: { x: 0, y: 0 }, end: { x: 50, y: 50 } }).events.length)
      .toBeGreaterThan(0);
    expect(await store.loadModel("missing")).toBeUndefined();
    expect(() => store.modelPath("../escape")).toThrow(/invalid model name/);
  });

  it("prunes episodes older than retainRawDays and removes emptied datasets", async () => {
    await consent.create({ tier: "app", purpose: "x" });
    const old = new Date(NOW.getTime() - 10 * 24 * 60 * 60 * 1000);
    await store.append([await drag("old", "drag-file", old), await drag("new", "drag-file", NOW, 2)]);
    await store.append([await drag("stale", "Close Tab", old, 3)]);

    const result = await store.prune();
    expect(result).toEqual({ removed: 2, datasets: 1 });
    expect((await store.readEpisodes("drag-file")).map((e) => e.id)).toEqual(["new"]);
    expect(fs.existsSync(store.datasetPath("Close Tab"))).toBe(false);
  });

  it("never deletes an unreadable dataset during prune", async () => {
    const file = store.datasetPath("drag-file");
    await fs.promises.mkdir(path.dirname(file), { recursive: true });
    await fs.promises.writeFile(file, "garbage\n");
    expect(await store.prune()).toEqual({ removed: 0, datasets: 0 });
    expect(fs.existsSync(file)).toBe(true);
  });

  it("gives colliding slugs distinct files", () => {
    expect(movementLabelFileStem("Drag File")).not.toBe(movementLabelFileStem("drag-file"));
    expect(movementLabelFileStem("../../etc")).toMatch(/^etc-[0-9a-f]{8}$/);
  });
});
