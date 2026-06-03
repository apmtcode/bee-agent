import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type PlatformBreakerState = "degraded" | "paused";

export type PlatformBreakerSource = "manual" | "automatic";

export type PlatformBreakerRecord = {
  platform: string;
  state: PlatformBreakerState;
  source: PlatformBreakerSource;
  reason?: string;
  failureCount?: number;
  threshold?: number;
  lastFailureAt?: string;
  lastFailureReason?: string;
  lastFailureRemoteId?: string;
  updatedAt: string;
};

export type PlatformFailureObservation = {
  platform: string;
  failureCount: number;
  lastFailureAt: string;
  lastFailureReason: string;
  lastFailureRemoteId?: string;
  lastFailureFingerprint: string;
  fingerprints: string[];
  updatedAt: string;
};

export type PlatformBreakerStoreShape = {
  version: 2;
  breakers: Record<string, PlatformBreakerRecord>;
  observations: Record<string, PlatformFailureObservation>;
};

type LegacyPlatformBreakerRecord = {
  platform: string;
  reason?: string;
  updatedAt: string;
};

type LegacyPlatformBreakerStoreShape = {
  version: 1;
  breakers: Record<string, LegacyPlatformBreakerRecord>;
};

const EMPTY_STORE: PlatformBreakerStoreShape = {
  version: 2,
  breakers: {},
  observations: {},
};

export class FilePlatformBreakerStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<PlatformBreakerStoreShape> {
    const raw = await readJsonFile<PlatformBreakerStoreShape | LegacyPlatformBreakerStoreShape>(this.filePath, EMPTY_STORE);
    return normalizeStore(raw);
  }

  async save(store: PlatformBreakerStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async listBreakers(): Promise<PlatformBreakerRecord[]> {
    const store = await this.load();
    return Object.values(store.breakers).sort((a, b) => a.platform.localeCompare(b.platform));
  }

  async getBreaker(platform: string): Promise<PlatformBreakerRecord | undefined> {
    const store = await this.load();
    return store.breakers[platform];
  }

  async setBreaker(params: {
    platform: string;
    reason?: string;
    updatedAt?: string;
    state?: PlatformBreakerState;
    source?: PlatformBreakerSource;
    failureCount?: number;
    threshold?: number;
    lastFailureAt?: string;
    lastFailureReason?: string;
    lastFailureRemoteId?: string;
  }): Promise<PlatformBreakerRecord> {
    const store = await this.load();
    const breaker: PlatformBreakerRecord = {
      platform: params.platform,
      state: params.state ?? "paused",
      source: params.source ?? "manual",
      ...(params.reason ? { reason: params.reason } : {}),
      ...(typeof params.failureCount === "number" ? { failureCount: params.failureCount } : {}),
      ...(typeof params.threshold === "number" ? { threshold: params.threshold } : {}),
      ...(params.lastFailureAt ? { lastFailureAt: params.lastFailureAt } : {}),
      ...(params.lastFailureReason ? { lastFailureReason: params.lastFailureReason } : {}),
      ...(params.lastFailureRemoteId ? { lastFailureRemoteId: params.lastFailureRemoteId } : {}),
      updatedAt: params.updatedAt ?? new Date().toISOString(),
    };
    store.breakers[params.platform] = breaker;
    if (breaker.source === "manual") {
      delete store.observations[params.platform];
    }
    await this.save(store);
    return breaker;
  }

  async recordAutomaticFailure(params: {
    platform: string;
    fingerprint: string;
    reason: string;
    remoteId?: string;
    degradeAfter: number;
    pauseAfter: number;
    updatedAt?: string;
  }): Promise<PlatformBreakerRecord | undefined> {
    const store = await this.load();
    const updatedAt = params.updatedAt ?? new Date().toISOString();
    const existingBreaker = store.breakers[params.platform];
    if (existingBreaker?.source === "manual") {
      return existingBreaker;
    }

    const existingObservation = store.observations[params.platform];
    const previousFingerprints = existingObservation?.fingerprints ?? (
      existingObservation?.lastFailureFingerprint
        ? [existingObservation.lastFailureFingerprint]
        : []
    );
    if (previousFingerprints.includes(params.fingerprint)) {
      return existingBreaker;
    }

    const failureCount = (existingObservation?.failureCount ?? 0) + 1;
    const observation: PlatformFailureObservation = {
      platform: params.platform,
      failureCount,
      lastFailureAt: updatedAt,
      lastFailureReason: params.reason,
      ...(params.remoteId ? { lastFailureRemoteId: params.remoteId } : {}),
      lastFailureFingerprint: params.fingerprint,
      fingerprints: [...previousFingerprints, params.fingerprint],
      updatedAt,
    };
    store.observations[params.platform] = observation;

    if (failureCount >= params.pauseAfter) {
      store.breakers[params.platform] = buildAutomaticBreakerRecord({
        platform: params.platform,
        state: "paused",
        reason: params.reason,
        remoteId: params.remoteId,
        failureCount,
        threshold: params.pauseAfter,
        updatedAt,
      });
    } else if (failureCount >= params.degradeAfter) {
      store.breakers[params.platform] = buildAutomaticBreakerRecord({
        platform: params.platform,
        state: "degraded",
        reason: params.reason,
        remoteId: params.remoteId,
        failureCount,
        threshold: params.degradeAfter,
        updatedAt,
      });
    }

    await this.save(store);
    return store.breakers[params.platform];
  }

  async clearBreaker(platform: string): Promise<boolean> {
    const store = await this.load();
    const hadBreaker = Boolean(store.breakers[platform]);
    const hadObservation = Boolean(store.observations[platform]);
    if (!hadBreaker && !hadObservation) {
      return false;
    }
    delete store.breakers[platform];
    delete store.observations[platform];
    await this.save(store);
    return true;
  }
}

function normalizeStore(raw: PlatformBreakerStoreShape | LegacyPlatformBreakerStoreShape): PlatformBreakerStoreShape {
  if (!raw || typeof raw !== "object") {
    return structuredClone(EMPTY_STORE);
  }
  if (raw.version === 2) {
    return {
      version: 2,
      breakers: { ...(raw.breakers ?? {}) },
      observations: Object.fromEntries(
        Object.entries(raw.observations ?? {}).map(([platform, observation]) => [platform, {
          ...observation,
          fingerprints: Array.isArray(observation?.fingerprints)
            ? observation.fingerprints.filter((item): item is string => typeof item === "string" && item.length > 0)
            : observation?.lastFailureFingerprint
              ? [observation.lastFailureFingerprint]
              : [],
        }]),
      ),
    };
  }
  const legacyBreakers = raw.version === 1 && raw.breakers && typeof raw.breakers === "object"
    ? Object.values(raw.breakers)
    : [];
  return {
    version: 2,
    breakers: Object.fromEntries(legacyBreakers.map((breaker) => [breaker.platform, {
      platform: breaker.platform,
      state: "paused",
      source: "manual",
      ...(breaker.reason ? { reason: breaker.reason } : {}),
      updatedAt: breaker.updatedAt,
    }])),
    observations: {},
  };
}

function buildAutomaticBreakerRecord(params: {
  platform: string;
  state: PlatformBreakerState;
  reason: string;
  remoteId?: string;
  failureCount: number;
  threshold: number;
  updatedAt: string;
}): PlatformBreakerRecord {
  return {
    platform: params.platform,
    state: params.state,
    source: "automatic",
    reason: `automatic retryable failures ${params.failureCount}/${params.threshold} ${params.reason}`,
    failureCount: params.failureCount,
    threshold: params.threshold,
    lastFailureAt: params.updatedAt,
    lastFailureReason: params.reason,
    ...(params.remoteId ? { lastFailureRemoteId: params.remoteId } : {}),
    updatedAt: params.updatedAt,
  };
}
