import path from "node:path";
import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type OperatorCliTeamRecord = {
  id: string;
  name: string;
  description?: string;
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

  async create(params: { name: string; description?: string }): Promise<OperatorCliTeamRecord> {
    const now = new Date().toISOString();
    const store = await this.read();
    const existing = store.teams.find((team) => team.name === params.name);
    if (existing) {
      return { ...existing };
    }
    const team: OperatorCliTeamRecord = {
      id: randomUUID(),
      name: params.name,
      ...(params.description ? { description: params.description } : {}),
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
