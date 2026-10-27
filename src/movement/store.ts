import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { isConsentGrantActive, type CaptureConsentGrant, type CapturePolicy, type CaptureTier } from "../capture/trajectory.js";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import {
  decodeMovementDataset,
  encodeMovementDataset,
  validateMovementEpisode,
  type MovementEpisode,
} from "./events.js";
import { TemplateMovementModel, type SerializedTemplateModel } from "./model.js";

/** Minimal consent lookup; `FileCaptureConsentStore` satisfies it. */
export type MovementConsentSource = {
  listActive(now?: Date): Promise<CaptureConsentGrant[]>;
};

export type FileMovementDatasetStoreOptions = {
  /** Root directory, e.g. `~/.openclaw/movement`. Datasets and models live beneath it. */
  rootDir: string;
  /** Recording requires an active grant at one of `recordingTiers`. */
  consent: MovementConsentSource;
  /** Capture policy; `allowTrainingExport` gates training reads, `retainRawDays` drives `prune`. */
  policy: () => CapturePolicy | Promise<CapturePolicy>;
  /** Tiers that permit raw input recording. Defaults to `app` and `full`. */
  recordingTiers?: CaptureTier[];
  now?: () => Date;
};

export type MovementDatasetSummary = {
  label: string;
  episodes: number;
  rejected: number;
  bytes: number;
  newestAt?: string;
};

export type AppendEpisodesResult = {
  grantId: string;
  appended: Array<{ id: string; label: string }>;
  rejected: Array<{ id: string; errors: string[] }>;
};

export class MovementConsentError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MovementConsentError";
  }
}

const DEFAULT_RECORDING_TIERS: CaptureTier[] = ["app", "full"];

/**
 * File name for a label: readable slug plus a short hash so distinct labels
 * that slugify identically ("Drag File" vs "drag-file") never share a file.
 */
export function movementLabelFileStem(label: string): string {
  const slug = label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  const hash = createHash("sha256").update(label).digest("hex").slice(0, 8);
  return `${slug || "label"}-${hash}`;
}

/**
 * Persists movement episodes as one JSONL dataset per label
 * (`<root>/datasets/<stem>.jsonl`) and model artifacts as JSON
 * (`<root>/models/<name>.json`). Every write is consent-gated, every training
 * read is gated by the policy's `allowTrainingExport`, and `prune` enforces
 * `retainRawDays`.
 */
export class FileMovementDatasetStore {
  private readonly rootDir: string;
  private readonly recordingTiers: CaptureTier[];
  private readonly now: () => Date;
  private writeQueue: Promise<unknown> = Promise.resolve();

  constructor(private readonly options: FileMovementDatasetStoreOptions) {
    this.rootDir = options.rootDir;
    this.recordingTiers = options.recordingTiers ?? DEFAULT_RECORDING_TIERS;
    this.now = options.now ?? (() => new Date());
  }

  datasetPath(label: string): string {
    return path.join(this.rootDir, "datasets", `${movementLabelFileStem(label)}.jsonl`);
  }

  modelPath(name: string): string {
    if (!/^[A-Za-z0-9._-]+$/.test(name) || name.startsWith(".")) {
      throw new Error(`invalid model name: ${name}`);
    }
    return path.join(this.rootDir, "models", `${name}.json`);
  }

  /** The active grant that authorizes recording, or a `MovementConsentError`. */
  async requireRecordingConsent(): Promise<CaptureConsentGrant> {
    const now = this.now();
    const grants = await this.options.consent.listActive(now);
    const grant = grants.find(
      (candidate) => isConsentGrantActive(candidate, now) && this.recordingTiers.includes(candidate.tier),
    );
    if (!grant) {
      throw new MovementConsentError(
        `movement recording requires an active consent grant at tier ${this.recordingTiers.join(" or ")}`,
      );
    }
    return grant;
  }

  /** Validate and append episodes to their label's dataset. Invalid episodes are rejected individually. */
  async append(episodes: MovementEpisode[]): Promise<AppendEpisodesResult> {
    const grant = await this.requireRecordingConsent();
    return await this.serialize(async () => {
      const result: AppendEpisodesResult = { grantId: grant.id, appended: [], rejected: [] };
      const byFile = new Map<string, MovementEpisode[]>();
      for (const episode of episodes) {
        const errors = validateMovementEpisode(episode);
        if (errors.length > 0) {
          result.rejected.push({ id: String((episode as { id?: unknown })?.id ?? "?"), errors });
          continue;
        }
        const file = this.datasetPath(episode.label);
        byFile.set(file, [...(byFile.get(file) ?? []), episode]);
        result.appended.push({ id: episode.id, label: episode.label });
      }
      for (const [file, batch] of byFile) {
        await fs.promises.mkdir(path.dirname(file), { recursive: true });
        const exists = await fileExists(file);
        const encoded = encodeMovementDataset(batch);
        // A new file gets the header line; existing files only get episode lines.
        const payload = exists ? encoded.slice(encoded.indexOf("\n") + 1) : encoded;
        await fs.promises.appendFile(file, payload, { encoding: "utf8", mode: 0o600 });
      }
      return result;
    });
  }

  async listDatasets(): Promise<MovementDatasetSummary[]> {
    const files = await this.datasetFiles();
    const summaries: MovementDatasetSummary[] = [];
    for (const file of files) {
      const raw = await fs.promises.readFile(file, "utf8");
      const decoded = safeDecode(raw);
      const label = decoded.episodes[0]?.label;
      if (!label) continue;
      const newestAt = decoded.episodes
        .map((episode) => episode.createdAt)
        .sort()
        .at(-1);
      summaries.push({
        label,
        episodes: decoded.episodes.length,
        rejected: decoded.rejected.length,
        bytes: Buffer.byteLength(raw),
        ...(newestAt ? { newestAt } : {}),
      });
    }
    return summaries.sort((a, b) => a.label.localeCompare(b.label));
  }

  /** Read one label's episodes for inspection (not gated: the user's own data, read locally). */
  async readEpisodes(label: string): Promise<MovementEpisode[]> {
    const raw = await readTextOrEmpty(this.datasetPath(label));
    return raw ? safeDecode(raw).episodes.filter((episode) => episode.label === label) : [];
  }

  /**
   * Episodes handed to a trainer. Gated by `allowTrainingExport` so captured
   * input never feeds a model unless the user opted in to training use.
   */
  async loadTrainingEpisodes(labels?: string[]): Promise<MovementEpisode[]> {
    const policy = await this.options.policy();
    if (!policy.allowTrainingExport) {
      throw new MovementConsentError("capture policy does not allow training export (allowTrainingExport=false)");
    }
    const wanted = labels ?? (await this.listDatasets()).map((summary) => summary.label);
    const episodes: MovementEpisode[] = [];
    for (const label of wanted) {
      episodes.push(...(await this.readEpisodes(label)));
    }
    return episodes;
  }

  /** Remove episodes older than `retainRawDays`; returns how many were dropped. */
  async prune(): Promise<{ removed: number; datasets: number }> {
    const policy = await this.options.policy();
    const cutoff = this.now().getTime() - policy.retainRawDays * 24 * 60 * 60 * 1000;
    return await this.serialize(async () => {
      let removed = 0;
      let datasets = 0;
      for (const file of await this.datasetFiles()) {
        let decoded: ReturnType<typeof decodeMovementDataset>;
        try {
          decoded = decodeMovementDataset(await fs.promises.readFile(file, "utf8"));
        } catch {
          // Never delete a file we cannot parse; leave it for a human to inspect.
          continue;
        }
        const kept = decoded.episodes.filter((episode) => new Date(episode.createdAt).getTime() >= cutoff);
        const dropped = decoded.episodes.length - kept.length;
        if (dropped === 0) {
          datasets += 1;
          continue;
        }
        removed += dropped;
        if (kept.length === 0) {
          await fs.promises.rm(file, { force: true });
          continue;
        }
        const temp = `${file}.${process.pid}.tmp`;
        await fs.promises.writeFile(temp, encodeMovementDataset(kept), { encoding: "utf8", mode: 0o600 });
        await fs.promises.rename(temp, file);
        datasets += 1;
      }
      return { removed, datasets };
    });
  }

  async saveModel(name: string, model: TemplateMovementModel): Promise<string> {
    const file = this.modelPath(name);
    await writeJsonAtomic(file, model.toJSON());
    return file;
  }

  async loadModel(name: string): Promise<TemplateMovementModel | undefined> {
    const data = await readJsonFile<SerializedTemplateModel | undefined>(this.modelPath(name), undefined);
    return data ? TemplateMovementModel.fromJSON(data) : undefined;
  }

  private async datasetFiles(): Promise<string[]> {
    const dir = path.join(this.rootDir, "datasets");
    try {
      const entries = await fs.promises.readdir(dir);
      return entries
        .filter((entry) => entry.endsWith(".jsonl"))
        .sort()
        .map((entry) => path.join(dir, entry));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw error;
    }
  }

  private serialize<T>(task: () => Promise<T>): Promise<T> {
    const next = this.writeQueue.then(task, task);
    this.writeQueue = next.catch(() => undefined);
    return next;
  }
}

function safeDecode(raw: string): ReturnType<typeof decodeMovementDataset> {
  try {
    return decodeMovementDataset(raw);
  } catch {
    return { episodes: [], rejected: [{ line: 1, errors: ["unreadable dataset"] }] };
  }
}

async function fileExists(file: string): Promise<boolean> {
  try {
    await fs.promises.access(file);
    return true;
  } catch {
    return false;
  }
}

async function readTextOrEmpty(file: string): Promise<string> {
  try {
    return await fs.promises.readFile(file, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return "";
    throw error;
  }
}
