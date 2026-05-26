import readline from "node:readline/promises";
import process from "node:process";
import { OperatorControlPlaneServer } from "../control-plane/server.js";
import { StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";
import { OperatorCliConfigLoader, type OperatorCliRuntimeConfig } from "./config.js";
import { discoverPromptContext, type OperatorCliPromptContext } from "./prompt.js";

export type OperatorCliSlashCommand =
  | { kind: "help" }
  | { kind: "status" }
  | { kind: "approvals" }
  | { kind: "recall"; query: string }
  | { kind: "skills" }
  | { kind: "run-skill"; skillId: string }
  | { kind: "background" }
  | { kind: "training" }
  | { kind: "unknown"; name: string };

export type OperatorCliAppOptions = {
  rootDir: string;
  cwd?: string;
  currentDate?: string;
  stdin?: NodeJS.ReadableStream;
  stdout?: NodeJS.WritableStream;
  stderr?: NodeJS.WritableStream;
};

export class OperatorCliApp {
  readonly runtime: StandaloneOperatorRuntime;
  readonly server: OperatorControlPlaneServer;
  readonly cwd: string;
  readonly currentDate: string;
  private readonly stdin: NodeJS.ReadableStream;
  private readonly stdout: NodeJS.WritableStream;
  private readonly stderr: NodeJS.WritableStream;

  constructor(private readonly options: OperatorCliAppOptions) {
    this.runtime = new StandaloneOperatorRuntime({ rootDir: options.rootDir });
    this.server = new OperatorControlPlaneServer({ runtime: this.runtime });
    this.cwd = options.cwd ?? process.cwd();
    this.currentDate = options.currentDate ?? new Date().toISOString().slice(0, 10);
    this.stdin = options.stdin ?? process.stdin;
    this.stdout = options.stdout ?? process.stdout;
    this.stderr = options.stderr ?? process.stderr;
  }

  async loadRuntimeConfig(): Promise<OperatorCliRuntimeConfig> {
    return await new OperatorCliConfigLoader(this.cwd).load();
  }

  async loadPromptContext(config?: OperatorCliRuntimeConfig): Promise<OperatorCliPromptContext> {
    return await discoverPromptContext({ cwd: this.cwd, currentDate: this.currentDate, config });
  }

  async runRepl(): Promise<void> {
    const config = await this.loadRuntimeConfig();
    const promptContext = await this.loadPromptContext(config);
    const session = await this.runtime.startSession({ title: "Operator CLI", cwd: this.cwd, agentId: "operator-cli" });
    this.stdout.write("Standalone Operator CLI\n");
    this.stdout.write(`Session: ${session.id}\n`);
    this.stdout.write(`Loaded config files: ${config.loadedEntries.length}\n`);
    this.stdout.write(`Loaded instruction files: ${promptContext.project.instructionFiles.length}\n`);
    this.stdout.write("Type /help for commands. Ctrl+D exits.\n");

    const reader = readline.createInterface({ input: this.stdin, output: this.stdout });
    try {
      while (true) {
        const line = await reader.question("operator> ");
        if (!line.trim()) {
          continue;
        }
        this.stdout.write(`${await this.handleInput(line, session.id, config, promptContext)}\n`);
      }
    } catch {
      this.stdout.write("\n");
    } finally {
      reader.close();
    }
  }

  async handleInput(
    input: string,
    sessionId: string,
    config?: OperatorCliRuntimeConfig,
    promptContext?: OperatorCliPromptContext,
  ): Promise<string> {
    const command = parseSlashCommand(input);
    if (!command) {
      await this.runtime.recordTurn({
        sessionId,
        userText: input,
        assistantText: "Interactive freeform turns are not implemented yet. Use slash commands.",
        trajectorySummary: `CLI freeform input: ${input}`,
      });
      return "Freeform chat is not implemented yet. Use /help.";
    }
    return await this.dispatchSlashCommand(command, sessionId, config, promptContext);
  }

  async dispatchSlashCommand(
    command: OperatorCliSlashCommand,
    sessionId: string,
    config?: OperatorCliRuntimeConfig,
    promptContext?: OperatorCliPromptContext,
  ): Promise<string> {
    switch (command.kind) {
      case "help":
        return [
          "Available commands:",
          "  /help",
          "  /status",
          "  /approvals",
          "  /recall <query>",
          "  /skills",
          "  /run-skill <id>",
          "  /background",
          "  /training",
        ].join("\n");
      case "status": {
        const session = await this.runtime.getSession(sessionId);
        const latestAssistant = await this.runtime.getLatestAssistantText(sessionId);
        const effectiveConfig = config ?? (await this.loadRuntimeConfig());
        const effectivePromptContext = promptContext ?? (await this.loadPromptContext(effectiveConfig));
        return [
          `session=${session?.id ?? sessionId}`,
          `status=${session?.status ?? "unknown"}`,
          `cwd=${this.cwd}`,
          `configFiles=${effectiveConfig.loadedEntries.length}`,
          `instructionFiles=${effectivePromptContext.project.instructionFiles.length}`,
          `latestAssistant=${latestAssistant ?? "<none>"}`,
        ].join(" ");
      }
      case "approvals": {
        const approvals = await this.runtime.listApprovals();
        return approvals.length === 0
          ? "No pending approvals."
          : approvals.map((approval) => `${approval.id} ${approval.title} (${approval.summary})`).join("\n");
      }
      case "recall": {
        const recall = await this.runtime.recall(command.query);
        if (recall.hits.length === 0) {
          return `No recall hits for \"${command.query}\".`;
        }
        return recall.hits
          .slice(0, 5)
          .map((hit) => {
            if (hit.kind === "memory") {
              return `memory ${hit.item.id}: ${hit.item.summary}`;
            }
            if (hit.kind === "skill") {
              return `skill ${hit.skill.id}: ${hit.skill.title}`;
            }
            return `trajectory ${hit.trajectoryId}: ${hit.summary}`;
          })
          .join("\n");
      }
      case "skills": {
        const promoted = await this.runtime.listPromotedSkills();
        const executable = await this.runtime.listExecutableSkills();
        if (promoted.length === 0 && executable.length === 0) {
          return "No promoted or executable skills.";
        }
        return [
          ...promoted.map((skill) => `promoted ${skill.id}: ${skill.title}`),
          ...executable.map((skill) => `executable ${skill.id}: ${skill.title}`),
        ].join("\n");
      }
      case "run-skill": {
        const run = await this.runtime.runExecutableSkill({ skillId: command.skillId, sessionId });
        if (!run) {
          return `Unknown executable skill: ${command.skillId}`;
        }
        return [`run ${run.id} completed`, ...run.stepResults.map((step) => `- ${step.stepId}: ${step.output}`)].join("\n");
      }
      case "background": {
        const tasks = await this.runtime.listBackgroundTasks(sessionId);
        return tasks.length === 0
          ? "No background tasks."
          : tasks.map((task) => `${task.id} ${task.status} ${task.title}`).join("\n");
      }
      case "training": {
        const jobs = await this.runtime.listTrainingJobs();
        return jobs.length === 0
          ? "No training jobs."
          : jobs.map((job) => `${job.id} ${job.status} ${job.mode}`).join("\n");
      }
      case "unknown":
        return `Unknown slash command: /${command.name}`;
    }
  }

  writeError(message: string): void {
    this.stderr.write(`${message}\n`);
  }
}

export function parseSlashCommand(input: string): OperatorCliSlashCommand | undefined {
  const trimmed = input.trim();
  if (!trimmed.startsWith("/")) {
    return undefined;
  }
  const [name, ...rest] = trimmed.slice(1).split(/\s+/);
  const tail = rest.join(" ").trim();
  switch (name) {
    case "help":
      return { kind: "help" };
    case "status":
      return { kind: "status" };
    case "approvals":
      return { kind: "approvals" };
    case "recall":
      return { kind: "recall", query: tail };
    case "skills":
      return { kind: "skills" };
    case "run-skill":
      return { kind: "run-skill", skillId: tail };
    case "background":
      return { kind: "background" };
    case "training":
      return { kind: "training" };
    default:
      return { kind: "unknown", name };
  }
}
