import { BrowserCaptureAdapter, type BrowserCaptureInput } from "./browser-adapter.js";
import { FileCaptureConsentStore } from "./consent-store.js";
import { DeviceCaptureAdapter, type DeviceCaptureInput } from "./device-adapter.js";
import { OsCaptureObserver, type OsObservationInput } from "./os-observer.js";
import {
  ConsentAwareCaptureRecorder,
  type CaptureRecordResult,
} from "./recorder.js";
import { FileTrajectoryStore } from "./trajectory-store.js";
import type { CaptureConsentGrant, CapturePolicy, CaptureTier } from "./trajectory.js";

export type CaptureIngestionServiceOptions = {
  consentStore: FileCaptureConsentStore;
  trajectoryStore: FileTrajectoryStore;
  policy?: CapturePolicy;
};

export type CreateCaptureConsentParams = {
  tier: CaptureTier;
  purpose: string;
  expiresAt?: string;
};

export class CaptureIngestionService {
  readonly recorder: ConsentAwareCaptureRecorder;
  readonly browser: BrowserCaptureAdapter;
  readonly device: DeviceCaptureAdapter;
  readonly os: OsCaptureObserver;

  constructor(private readonly options: CaptureIngestionServiceOptions) {
    this.recorder = new ConsentAwareCaptureRecorder(options);
    this.browser = new BrowserCaptureAdapter(this.recorder);
    this.device = new DeviceCaptureAdapter(this.recorder);
    this.os = new OsCaptureObserver(this.recorder);
  }

  async createConsent(params: CreateCaptureConsentParams): Promise<CaptureConsentGrant> {
    return await this.options.consentStore.create(params);
  }

  async listActiveConsents(now: Date = new Date()): Promise<CaptureConsentGrant[]> {
    return await this.options.consentStore.listActive(now);
  }

  async revokeConsent(grantId: string): Promise<CaptureConsentGrant | undefined> {
    return await this.options.consentStore.revoke(grantId);
  }

  async recordBrowser(input: BrowserCaptureInput): Promise<CaptureRecordResult> {
    return await this.browser.record(input);
  }

  async recordDevice(input: DeviceCaptureInput): Promise<CaptureRecordResult> {
    return await this.device.record(input);
  }

  async observeOs(input: OsObservationInput): Promise<CaptureRecordResult> {
    return await this.os.observe(input);
  }
}
