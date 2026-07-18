import {
  MovementModel,
  trainMovementModel,
  type MovementDataset,
  type MovementModelFidelity,
  type MovementSequence,
  type MovementToken,
  type SerializedMovementModel,
  type TrainMovementModelOptions,
} from "./movement-model.js";

/**
 * Serialized, backend-agnostic training artifact. Native backends (e.g. an
 * on-device mlx/axolotl runner) can set `format`/`nativeArtifactPath` and omit
 * the in-process `model`, while the deterministic mock backend embeds a fully
 * replayable `model` so cloud/CI can train and infer end-to-end.
 */
export type MovementModelArtifact = {
  version: 1;
  backendId: string;
  format: string;
  createdAt?: string;
  sequenceCount: number;
  actionCount: number;
  /** Present for in-process backends; a native backend points at a file instead. */
  model?: SerializedMovementModel;
  nativeArtifactPath?: string;
  metadata?: Record<string, unknown>;
};

export type TrainMovementBackendOptions = TrainMovementModelOptions & {
  createdAt?: string;
  metadata?: Record<string, unknown>;
};

export type MovementInferenceRequest = {
  /** Seed context (usually the observations describing a new situation). */
  seed?: MovementToken[];
  maxSteps?: number;
};

/**
 * A loaded, ready-to-infer movement model. This is the runtime seam a caller
 * uses regardless of which backend produced the artifact.
 */
export interface MovementModelHandle {
  readonly backendId: string;
  /** Repeat/generalize: roll out an action sequence from a seed context. */
  infer(request?: MovementInferenceRequest): Extract<MovementToken, { role: "action" }>[];
  /** Teacher-forced fidelity against a held-out related sequence. */
  evaluate(sequence: MovementSequence): MovementModelFidelity;
  serialize(): MovementModelArtifact;
}

/**
 * Pluggable training backend. Implementations turn a reviewed movement dataset
 * into an artifact and can reload that artifact for inference. The in-process
 * mock is the reference implementation; native/on-device backends implement the
 * same contract so the rest of bee-agent is backend-agnostic.
 */
export interface MovementTrainingBackend {
  readonly id: string;
  readonly kind: "in-process" | "native-deferred";
  train(dataset: MovementDataset, options?: TrainMovementBackendOptions): Promise<MovementModelArtifact>;
  load(artifact: MovementModelArtifact): MovementModelHandle;
}

function countActions(dataset: MovementDataset): number {
  return dataset.sequences.reduce(
    (total, sequence) => total + sequence.tokens.filter((token) => token.role === "action").length,
    0,
  );
}

/**
 * Deterministic, dependency-free backend that trains the in-process
 * {@link MovementModel}. Runs anywhere (including the cloud), so it validates
 * the capture → dataset → train → infer pipeline without real OS access.
 */
export class MockMovementTrainingBackend implements MovementTrainingBackend {
  readonly id = "mock";
  readonly kind = "in-process" as const;

  async train(
    dataset: MovementDataset,
    options: TrainMovementBackendOptions = {},
  ): Promise<MovementModelArtifact> {
    const model = trainMovementModel(dataset, options);
    return {
      version: 1,
      backendId: this.id,
      format: "movement-ngram/v1",
      ...(options.createdAt ? { createdAt: options.createdAt } : {}),
      sequenceCount: dataset.sequences.length,
      actionCount: countActions(dataset),
      model: model.toJSON(),
      ...(options.metadata ? { metadata: options.metadata } : {}),
    };
  }

  load(artifact: MovementModelArtifact): MovementModelHandle {
    if (!artifact.model) {
      throw new Error(`Artifact for backend ${artifact.backendId} has no in-process model to load`);
    }
    const model = MovementModel.fromJSON(artifact.model);
    return new InProcessMovementModelHandle(this.id, model);
  }
}

class InProcessMovementModelHandle implements MovementModelHandle {
  constructor(
    readonly backendId: string,
    private readonly model: MovementModel,
  ) {}

  infer(request: MovementInferenceRequest = {}): Extract<MovementToken, { role: "action" }>[] {
    return this.model.generate(request.seed ?? [], request.maxSteps ?? 64);
  }

  evaluate(sequence: MovementSequence): MovementModelFidelity {
    return this.model.evaluateFidelity(sequence.tokens);
  }

  serialize(): MovementModelArtifact {
    return {
      version: 1,
      backendId: this.backendId,
      format: "movement-ngram/v1",
      sequenceCount: 0,
      actionCount: 0,
      model: this.model.toJSON(),
    };
  }
}

/**
 * Registry that makes the training backend pluggable: register native or
 * experimental backends by id and resolve the right one for a given artifact.
 */
export class MovementTrainingBackendRegistry {
  private readonly backends = new Map<string, MovementTrainingBackend>();

  register(backend: MovementTrainingBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  get(id: string): MovementTrainingBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`Unknown movement training backend: ${id}`);
    }
    return backend;
  }

  list(): MovementTrainingBackend[] {
    return [...this.backends.values()];
  }

  /** Load an artifact using the backend that produced it. */
  load(artifact: MovementModelArtifact): MovementModelHandle {
    return this.get(artifact.backendId).load(artifact);
  }
}

/** A registry seeded with the deterministic in-process mock backend. */
export function createDefaultMovementBackendRegistry(): MovementTrainingBackendRegistry {
  return new MovementTrainingBackendRegistry().register(new MockMovementTrainingBackend());
}
