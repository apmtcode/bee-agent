import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type PlatformBreakerRecord = {
  platform: string;
  reason?: string;
  updatedAt: string;
};

export type PlatformBreakerStoreShape = {
  version: 1;
  breakers: Record<string, PlatformBreakerRecord>;
};

const EMPTY_STORE: PlatformBreakerStoreShape = {
  version: 1,
  breakers: {},
};

export class FilePlatformBreakerStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<PlatformBreakerStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
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

  async setBreaker(params: { platform: string; reason?: string; updatedAt?: string }): Promise<PlatformBreakerRecord> {
    const store = await this.load();
    const breaker: PlatformBreakerRecord = {
      platform: params.platform,
      ...(params.reason ? { reason: params.reason } : {}),
      updatedAt: params.updatedAt ?? new Date().toISOString(),
    };
    store.breakers[params.platform] = breaker;
    await this.save(store);
    return breaker;
  }

  async clearBreaker(platform: string): Promise<boolean> {
    const store = await this.load();
    if (!store.breakers[platform]) {
      return false;
    }
    delete store.breakers[platform];
    await this.save(store);
    return true;
  }
}
