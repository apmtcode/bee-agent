import path from "node:path";
import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type OperatorCliTeamRecord = {
  id: string;
  name: string;
  description?: string;
  taskSessionId?: string;
  createdAt: string;
  updatedAt: string;
};

type OperatorCliTeamStoreShape = {
  teams: OperatorCliTeamRecord[];
};

const EMPTY_STORE: OperatorCliTeamStoreShape = { teams: [] };

export class FileOperatorCliTeamStore {
  private readonly filePath: string;

  constructor(rootDir: string) {
    this.filePath = path.join(rootDir, "operator-cli-teams.json");
  }

  async list(): Promise<OperatorCliTeamRecord[]> {
    const store = await this.read();
    return store.teams.map((team) => ({ ...team }));
  }

  async get(name: string): Promise<OperatorCliTeamRecord | undefined> {
    const store = await this.read();
    const team = store.teams.find((item) => item.name === name);
    return team ? { ...team } : undefined;
  }

  async create(params: { name: string; description?: string; taskSessionId?: string }): Promise<OperatorCliTeamRecord> {
    const now = new Date().toISOString();
    const store = await this.read();
    const existingIndex = store.teams.findIndex((team) => team.name === params.name);
    if (existingIndex >= 0) {
      const existing = store.teams[existingIndex];
      if (params.taskSessionId && !existing.taskSessionId) {
        const updated = {
          ...existing,
          taskSessionId: params.taskSessionId,
          updatedAt: now,
        };
        store.teams[existingIndex] = updated;
        await this.write(store);
        return { ...updated };
      }
      return { ...existing };
    }
    const team: OperatorCliTeamRecord = {
      id: randomUUID(),
      name: params.name,
      ...(params.description ? { description: params.description } : {}),
      ...(params.taskSessionId ? { taskSessionId: params.taskSessionId } : {}),
      createdAt: now,
      updatedAt: now,
    };
    store.teams.push(team);
    await this.write(store);
    return { ...team };
  }

  async delete(name: string): Promise<OperatorCliTeamRecord | undefined> {
    const store = await this.read();
    const index = store.teams.findIndex((team) => team.name === name);
    if (index < 0) {
      return undefined;
    }
    const [team] = store.teams.splice(index, 1);
    await this.write(store);
    return team ? { ...team } : undefined;
  }

  private async read(): Promise<OperatorCliTeamStoreShape> {
    const store = await readJsonFile<OperatorCliTeamStoreShape>(this.filePath, EMPTY_STORE);
    return {
      teams: Array.isArray(store.teams) ? store.teams.map((team) => ({ ...team })) : [],
    };
  }

  private async write(store: OperatorCliTeamStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }
}
