import type { MovementDataset } from "./movement-dataset.js";

/**
 * Pluggable local-model backend seam for the movement learning subsystem.
 *
 * bee-agent runs in the cloud and cannot train on a real device, so training is
 * expressed against this interface. The default {@link NgramMovementBackend}
 * (see `ngram-backend.ts`) is a deterministic, dependency-free reference model
 * that trains and infers fully in-process — good for tests and offline use. A
 * real on-device backend (e.g. MLX/LoRA fine-tuning of a small local model) can
 * implement the same interface and be registered under its own id, so callers
 * are decoupled from which model actually runs.
 */

export type MovementTrainingConfig = {
  /** Maximum context length the model conditions on (n-gram order = order + 1). */
  order: number;
  /** Drop contexts observed fewer than this many times. Defaults to 1. */
  minCount?: number;
};

export type MovementPrediction = {
  symbol: string;
  probability: number;
  /** Context length actually used after backoff (0 = unconditional). */
  order: number;
};

export type SerializedMovementModel = {
  backendId: string;
  version: 1;
  order: number;
  vocabulary: string[];
  grams: Array<{
    context: string[];
    next: Array<{ symbol: string; count: number }>;
  }>;
};

export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Most likely next symbol given a context, or undefined if nothing is known. */
  predict(context: string[]): MovementPrediction | undefined;
  /** Full next-symbol distribution (descending probability, deterministic ties). */
  distribution(context: string[]): MovementPrediction[];
  /**
   * Roll out a movement sequence from a seed context, stopping at the end
   * sentinel or `maxLength` tokens. Deterministic: always takes the argmax.
   */
  generate(seed?: string[], options?: { maxLength?: number }): string[];
  serialize(): SerializedMovementModel;
}

export interface LocalMovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config: MovementTrainingConfig): Promise<TrainedMovementModel>;
  load(serialized: SerializedMovementModel): TrainedMovementModel;
}

/** Registry so callers can select a training backend by id at runtime. */
export class MovementModelBackendRegistry {
  private readonly backends = new Map<string, LocalMovementModelBackend>();

  register(backend: LocalMovementModelBackend): this {
    if (this.backends.has(backend.id)) {
      throw new Error(`movement backend already registered: ${backend.id}`);
    }
    this.backends.set(backend.id, backend);
    return this;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  get(id: string): LocalMovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement backend: ${id} (registered: ${this.list().join(", ") || "none"})`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}
