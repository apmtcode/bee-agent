import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import { isConsentGrantActive, type CaptureConsentGrant, type CaptureTier } from "./trajectory.js";

export type CaptureConsentStoreShape = {
  version: 1;
  grants: CaptureConsentGrant[];
};

const EMPTY_STORE: CaptureConsentStoreShape = {
  version: 1,
  grants: [],
};

function nowIso(): string {
  return new Date().toISOString();
}

export class FileCaptureConsentStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<CaptureConsentStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: CaptureConsentStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async list(): Promise<CaptureConsentGrant[]> {
    const store = await this.load();
    return [...store.grants].sort((a, b) => a.grantedAt.localeCompare(b.grantedAt));
  }

  async get(grantId: string): Promise<CaptureConsentGrant | undefined> {
    const store = await this.load();
    return store.grants.find((grant) => grant.id === grantId);
  }

  async create(params: {
    tier: CaptureTier;
    purpose: string;
    expiresAt?: string;
  }): Promise<CaptureConsentGrant> {
    const store = await this.load();
    const grant: CaptureConsentGrant = {
      id: randomUUID(),
      tier: params.tier,
      grantedAt: nowIso(),
      purpose: params.purpose,
      ...(params.expiresAt ? { expiresAt: params.expiresAt } : {}),
    };
    store.grants.push(grant);
    await this.save(store);
    return grant;
  }

  async revoke(grantId: string): Promise<CaptureConsentGrant | undefined> {
    const store = await this.load();
    const index = store.grants.findIndex((grant) => grant.id === grantId);
    if (index < 0) {
      return undefined;
    }
    const revoked: CaptureConsentGrant = {
      ...store.grants[index],
      revokedAt: nowIso(),
    };
    store.grants[index] = revoked;
    await this.save(store);
    return revoked;
  }

  async listActive(now: Date = new Date()): Promise<CaptureConsentGrant[]> {
    const grants = await this.list();
    return grants.filter((grant) => isConsentGrantActive(grant, now));
  }

  async findActiveByTier(tier: CaptureTier, now: Date = new Date()): Promise<CaptureConsentGrant | undefined> {
    const grants = await this.listActive(now);
    return grants.find((grant) => grant.tier === tier);
  }
}
