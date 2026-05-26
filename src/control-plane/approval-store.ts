import type { ApprovalRequest, ApprovalResolution } from "./approvals.js";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type ApprovalStoreShape = {
  version: 1;
  pending: ApprovalRequest[];
  resolutions: ApprovalResolution[];
};

const EMPTY_STORE: ApprovalStoreShape = {
  version: 1,
  pending: [],
  resolutions: [],
};

export class FileApprovalStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<ApprovalStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: ApprovalStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }
}
