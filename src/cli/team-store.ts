import path from "node:path";
import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type OperatorCliTeammateStatus = "pending" | "running" | "paused" | "completed" | "failed";

export type OperatorCliTeammateRecord = {
  id: string;
  name: string;
  sessionId: string;
  subagentRunId: string;
  childRunId: string;
  title: string;
  agentId?: string;
  createdAt: string;
  updatedAt: string;
  status: OperatorCliTeammateStatus;
};

export type OperatorCliTeamRecord = {
  id: string;
  name: string;
  description?: string;
  taskSessionId?: string;
  teammates: OperatorCliTeammateRecord[];
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
    return store.teams.map((team) => this.cloneTeam(team));
  }

  async get(name: string): Promise<OperatorCliTeamRecord | undefined> {
    const store = await this.read();
    const team = store.teams.find((item) => item.name === name);
    return team ? this.cloneTeam(team) : undefined;
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
        } satisfies OperatorCliTeamRecord;
        store.teams[existingIndex] = updated;
        await this.write(store);
        return this.cloneTeam(updated);
      }
      return this.cloneTeam(existing);
    }
    const team: OperatorCliTeamRecord = {
      id: randomUUID(),
      name: params.name,
      ...(params.description ? { description: params.description } : {}),
      ...(params.taskSessionId ? { taskSessionId: params.taskSessionId } : {}),
      teammates: [],
      createdAt: now,
      updatedAt: now,
    };
    store.teams.push(team);
    await this.write(store);
    return this.cloneTeam(team);
  }

  async delete(name: string): Promise<OperatorCliTeamRecord | undefined> {
    const store = await this.read();
    const index = store.teams.findIndex((team) => team.name === name);
    if (index < 0) {
      return undefined;
    }
    const [team] = store.teams.splice(index, 1);
    await this.write(store);
    return team ? this.cloneTeam(team) : undefined;
  }

  async listTeammates(teamName: string): Promise<OperatorCliTeammateRecord[]> {
    const team = await this.get(teamName);
    return team ? team.teammates.map((teammate) => this.cloneTeammate(teammate)) : [];
  }

  async getTeammate(teamName: string, teammateName: string): Promise<OperatorCliTeammateRecord | undefined> {
    const team = await this.get(teamName);
    const teammate = team?.teammates.find((item) => item.name === teammateName);
    return teammate ? this.cloneTeammate(teammate) : undefined;
  }

  async addTeammate(params: {
    teamName: string;
    teammateName: string;
    sessionId: string;
    subagentRunId: string;
    childRunId: string;
    title: string;
    agentId?: string;
    status?: OperatorCliTeammateStatus;
  }): Promise<OperatorCliTeammateRecord> {
    const now = new Date().toISOString();
    const store = await this.read();
    const teamIndex = store.teams.findIndex((team) => team.name === params.teamName);
    if (teamIndex < 0) {
      throw new Error(`unknown team: ${params.teamName}`);
    }
    const team = store.teams[teamIndex];
    if (team.teammates.some((teammate) => teammate.name === params.teammateName)) {
      throw new Error(`duplicate teammate: ${params.teamName}/${params.teammateName}`);
    }
    const teammate: OperatorCliTeammateRecord = {
      id: randomUUID(),
      name: params.teammateName,
      sessionId: params.sessionId,
      subagentRunId: params.subagentRunId,
      childRunId: params.childRunId,
      title: params.title,
      ...(params.agentId ? { agentId: params.agentId } : {}),
      createdAt: now,
      updatedAt: now,
      status: params.status ?? "running",
    };
    team.teammates.push(teammate);
    team.updatedAt = now;
    store.teams[teamIndex] = team;
    await this.write(store);
    return this.cloneTeammate(teammate);
  }

  async updateTeammate(
    teamName: string,
    teammateName: string,
    params: {
      sessionId?: string;
      subagentRunId?: string;
      childRunId?: string;
      title?: string;
      agentId?: string | null;
      status?: OperatorCliTeammateStatus;
    },
  ): Promise<OperatorCliTeammateRecord | undefined> {
    const now = new Date().toISOString();
    const store = await this.read();
    const teamIndex = store.teams.findIndex((team) => team.name === teamName);
    if (teamIndex < 0) {
      return undefined;
    }
    const team = store.teams[teamIndex];
    const teammateIndex = team.teammates.findIndex((teammate) => teammate.name === teammateName);
    if (teammateIndex < 0) {
      return undefined;
    }
    const current = team.teammates[teammateIndex];
    const updated: OperatorCliTeammateRecord = {
      ...current,
      ...(params.sessionId ? { sessionId: params.sessionId } : {}),
      ...(params.subagentRunId ? { subagentRunId: params.subagentRunId } : {}),
      ...(params.childRunId ? { childRunId: params.childRunId } : {}),
      ...(params.title ? { title: params.title } : {}),
      ...(params.agentId === null ? {} : params.agentId ? { agentId: params.agentId } : current.agentId ? { agentId: current.agentId } : {}),
      ...(params.status ? { status: params.status } : {}),
      updatedAt: now,
    };
    if (params.agentId === null) {
      delete updated.agentId;
    }
    team.teammates[teammateIndex] = updated;
    team.updatedAt = now;
    store.teams[teamIndex] = team;
    await this.write(store);
    return this.cloneTeammate(updated);
  }

  async removeTeammate(teamName: string, teammateName: string): Promise<OperatorCliTeammateRecord | undefined> {
    const now = new Date().toISOString();
    const store = await this.read();
    const teamIndex = store.teams.findIndex((team) => team.name === teamName);
    if (teamIndex < 0) {
      return undefined;
    }
    const team = store.teams[teamIndex];
    const teammateIndex = team.teammates.findIndex((teammate) => teammate.name === teammateName);
    if (teammateIndex < 0) {
      return undefined;
    }
    const [teammate] = team.teammates.splice(teammateIndex, 1);
    team.updatedAt = now;
    store.teams[teamIndex] = team;
    await this.write(store);
    return teammate ? this.cloneTeammate(teammate) : undefined;
  }

  private async read(): Promise<OperatorCliTeamStoreShape> {
    const store = await readJsonFile<OperatorCliTeamStoreShape>(this.filePath, EMPTY_STORE);
    return {
      teams: Array.isArray(store.teams) ? store.teams.map((team) => this.normalizeTeam(team)) : [],
    };
  }

  private async write(store: OperatorCliTeamStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  private normalizeTeam(team: Partial<OperatorCliTeamRecord>): OperatorCliTeamRecord {
    return {
      id: typeof team.id === "string" ? team.id : randomUUID(),
      name: typeof team.name === "string" ? team.name : "",
      ...(typeof team.description === "string" ? { description: team.description } : {}),
      ...(typeof team.taskSessionId === "string" ? { taskSessionId: team.taskSessionId } : {}),
      teammates: Array.isArray(team.teammates)
        ? team.teammates.map((teammate) => this.normalizeTeammate(teammate))
        : [],
      createdAt: typeof team.createdAt === "string" ? team.createdAt : new Date(0).toISOString(),
      updatedAt: typeof team.updatedAt === "string" ? team.updatedAt : new Date(0).toISOString(),
    };
  }

  private normalizeTeammate(teammate: Partial<OperatorCliTeammateRecord>): OperatorCliTeammateRecord {
    return {
      id: typeof teammate.id === "string" ? teammate.id : randomUUID(),
      name: typeof teammate.name === "string" ? teammate.name : "",
      sessionId: typeof teammate.sessionId === "string" ? teammate.sessionId : "",
      subagentRunId: typeof teammate.subagentRunId === "string" ? teammate.subagentRunId : "",
      childRunId: typeof teammate.childRunId === "string" ? teammate.childRunId : "",
      title: typeof teammate.title === "string" ? teammate.title : "",
      ...(typeof teammate.agentId === "string" ? { agentId: teammate.agentId } : {}),
      createdAt: typeof teammate.createdAt === "string" ? teammate.createdAt : new Date(0).toISOString(),
      updatedAt: typeof teammate.updatedAt === "string" ? teammate.updatedAt : new Date(0).toISOString(),
      status: teammate.status === "pending"
        || teammate.status === "running"
        || teammate.status === "paused"
        || teammate.status === "completed"
        || teammate.status === "failed"
        ? teammate.status
        : "pending",
    };
  }

  private cloneTeam(team: OperatorCliTeamRecord): OperatorCliTeamRecord {
    return {
      ...team,
      teammates: team.teammates.map((teammate) => this.cloneTeammate(teammate)),
    };
  }

  private cloneTeammate(teammate: OperatorCliTeammateRecord): OperatorCliTeammateRecord {
    return { ...teammate };
  }
}
