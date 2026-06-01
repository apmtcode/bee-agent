import { describe, expect, it } from "vitest";
import { resolveOperatorCliExecutionConfig } from "./config.js";
import { runOperatorHooks } from "./execution-policy.js";

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
