import { describe, expect, it } from "vitest";
import { resolveOperatorCliExecutionConfig } from "./config.js";
import { runOperatorHooks, runOperatorPreToolUseHooks } from "./execution-policy.js";

function buildEnvDumpCommand(): string {
  const script = [
    "const payload = {",
    "  event: process.env.OPERATOR_HOOK_EVENT,",
    "  cwd: process.env.OPERATOR_CWD,",
    "  permissionMode: process.env.OPERATOR_PERMISSION_MODE,",
    "  sessionId: process.env.OPERATOR_SESSION_ID,",
    "  runId: process.env.OPERATOR_RUN_ID,",
    "  approvalId: process.env.OPERATOR_APPROVAL_ID,",
    "  input: JSON.parse(process.env.OPERATOR_HOOK_INPUT || '{}'),",
    "};",
    "process.stdout.write(JSON.stringify(payload));",
  ].join(" ");
  return `node -e ${JSON.stringify(script)}`;
}

describe("runOperatorHooks", () => {
  it("passes generic lifecycle hook context through the environment", async () => {
    const config = resolveOperatorCliExecutionConfig({
      permissionMode: "acceptEdits",
      hooks: {
        SessionStart: [buildEnvDumpCommand()],
      },
    });

    const results = await runOperatorHooks({
      config,
      context: {
        event: "SessionStart",
        cwd: process.cwd(),
        permissionMode: config.permissionMode,
        sessionId: "sess-1",
        runId: "run-1",
        approvalId: "approval-1",
        input: {
          session: {
            id: "sess-1",
            status: "active",
          },
        },
      },
    });

    expect(results).toHaveLength(1);
    const [result] = results;
    if (!result) {
      throw new Error("expected hook result");
    }
    expect(result.event).toBe("SessionStart");
    expect(JSON.parse(result.stdout)).toEqual({
      event: "SessionStart",
      cwd: process.cwd(),
      permissionMode: "acceptEdits",
      sessionId: "sess-1",
      runId: "run-1",
      approvalId: "approval-1",
      input: {
        session: {
          id: "sess-1",
          status: "active",
        },
      },
    });
  });
});

describe("runOperatorPreToolUseHooks", () => {
  it("treats non-json stdout as a no-op", async () => {
    const config = resolveOperatorCliExecutionConfig({
      hooks: {
        PreToolUse: ["printf 'plain output'"] ,
      },
    });

    const result = await runOperatorPreToolUseHooks({
      config,
      action: {
        kind: "background.start",
        cwd: process.cwd(),
        command: "printf ok",
        source: "cli",
      },
    });

    expect(result.outcome).toBe("allow");
    expect(result.action.command).toBe("printf ok");
    expect(result.results[0]?.parsed).toBeUndefined();
  });

  it("parses the last stdout line json and applies the last updated input", async () => {
    const config = resolveOperatorCliExecutionConfig({
      hooks: {
        PreToolUse: [
          "printf 'noise\\n{\"updatedInput\":{\"command\":\"printf first\",\"title\":\"First\"}}'",
          "printf '{\"updatedInput\":{\"command\":\"printf second\",\"title\":\"Second\"}}'",
        ],
      },
    });

    const result = await runOperatorPreToolUseHooks({
      config,
      action: {
        kind: "background.start",
        cwd: process.cwd(),
        command: "printf ok",
        title: "Original",
        source: "cli",
      },
    });

    expect(result.outcome).toBe("allow");
    expect(result.action.command).toBe("printf second");
    expect(result.action.title).toBe("Second");
  });

  it("gives deny precedence over request-approval", async () => {
    const config = resolveOperatorCliExecutionConfig({
      hooks: {
        PreToolUse: [
          "printf '{\"permissionDecision\":\"request-approval\",\"permissionDecisionReason\":\"needs review\"}'",
          "printf '{\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"blocked\"}'",
        ],
      },
    });

    const result = await runOperatorPreToolUseHooks({
      config,
      action: {
        kind: "background.start",
        cwd: process.cwd(),
        command: "printf ok",
        source: "cli",
      },
    });

    expect(result.outcome).toBe("deny");
    expect(result.reason).toBe("blocked");
  });

  it("preserves request-approval when no deny is present", async () => {
    const config = resolveOperatorCliExecutionConfig({
      hooks: {
        PreToolUse: [
          "printf '{\"permissionDecision\":\"allow\"}'",
          "printf '{\"permissionDecision\":\"request-approval\",\"permissionDecisionReason\":\"needs review\",\"additionalContext\":\"explain why\"}'",
        ],
      },
    });

    const result = await runOperatorPreToolUseHooks({
      config,
      action: {
        kind: "background.start",
        cwd: process.cwd(),
        command: "printf ok",
        source: "cli",
      },
    });

    expect(result.outcome).toBe("request-approval");
    expect(result.reason).toBe("needs review");
    expect(result.additionalContext).toBe("explain why");
  });
});
