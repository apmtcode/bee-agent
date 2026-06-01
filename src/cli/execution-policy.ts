import { execFile } from "node:child_process";
import process from "node:process";
import { promisify } from "node:util";
import type { OperatorCliExecutionConfig, OperatorCliHookEvent, OperatorCliPermissionMode } from "./config.js";

const execFileAsync = promisify(execFile);

export type OperatorExecutionActionKind = "background.start" | "skill.command";

export type OperatorExecutionAction = {
  kind: OperatorExecutionActionKind;
  sessionId?: string;
  cwd: string;
  command: string;
  title?: string;
  source: "cli" | "runtime" | "rpc";
};

export type OperatorExecutionPolicyDecision = {
  outcome: "allow" | "deny" | "request-approval";
  permissionMode: OperatorCliPermissionMode;
  fingerprint: string;
  dangerousReasons: string[];
  reason: string;
};

export type OperatorHookResult = {
  event: OperatorCliHookEvent;
  command: string;
  stdout: string;
  stderr: string;
};

export type OperatorHookContext = {
  event: OperatorCliHookEvent;
  cwd: string;
  permissionMode: OperatorCliPermissionMode;
  sessionId?: string;
  runId?: string;
  approvalId?: string;
  actionKind?: OperatorExecutionActionKind;
  command?: string;
  title?: string;
  input?: Record<string, unknown>;
};

export function evaluateOperatorExecutionAction(params: {
  action: OperatorExecutionAction;
  config: OperatorCliExecutionConfig;
  approvedFingerprints?: ReadonlySet<string>;
}): OperatorExecutionPolicyDecision {
  const fingerprint = buildOperatorExecutionFingerprint(params.action);
  const dangerousReasons = detectDangerousCommandReasons(params.action.command);

  if (params.config.invalidPermissionMode) {
    return {
      outcome: "deny",
      permissionMode: params.config.permissionMode,
      fingerprint,
      dangerousReasons,
      reason: `Invalid permissionMode: ${params.config.invalidPermissionMode}`,
    };
  }

  if (params.approvedFingerprints?.has(fingerprint)) {
    return {
      outcome: "allow",
      permissionMode: params.config.permissionMode,
      fingerprint,
      dangerousReasons,
      reason: "Previously approved in this session.",
    };
  }

  if (params.config.permissionMode === "plan") {
    return {
      outcome: "deny",
      permissionMode: params.config.permissionMode,
      fingerprint,
      dangerousReasons,
      reason: "Execution is disabled while permissionMode=plan.",
    };
  }

  if (params.config.permissionMode === "bypassPermissions") {
    return {
      outcome: "allow",
      permissionMode: params.config.permissionMode,
      fingerprint,
      dangerousReasons,
      reason: "Allowed by bypassPermissions.",
    };
  }

  if (dangerousReasons.length > 0) {
    return {
      outcome: "request-approval",
      permissionMode: params.config.permissionMode,
      fingerprint,
      dangerousReasons,
      reason: dangerousReasons.join("; "),
    };
  }

  return {
    outcome: "allow",
    permissionMode: params.config.permissionMode,
    fingerprint,
    dangerousReasons,
    reason: "Allowed by policy.",
  };
}

export function buildOperatorExecutionFingerprint(action: OperatorExecutionAction): string {
  return [action.kind, action.cwd, action.command.trim()].join("|");
}

export async function runOperatorHooks(params: {
  config: OperatorCliExecutionConfig;
  context: OperatorHookContext;
}): Promise<OperatorHookResult[]> {
  const commands = params.config.hooks[params.context.event];
  const results: OperatorHookResult[] = [];
  for (const hookCommand of commands) {
    const { stdout, stderr } = await execFileAsync("bash", ["-lc", hookCommand], {
      cwd: params.context.cwd,
      env: {
        ...process.env,
        OPERATOR_HOOK_EVENT: params.context.event,
        OPERATOR_CWD: params.context.cwd,
        OPERATOR_PERMISSION_MODE: params.context.permissionMode,
        ...(params.context.sessionId ? { OPERATOR_SESSION_ID: params.context.sessionId } : {}),
        ...(params.context.runId ? { OPERATOR_RUN_ID: params.context.runId } : {}),
        ...(params.context.approvalId ? { OPERATOR_APPROVAL_ID: params.context.approvalId } : {}),
        ...(params.context.actionKind ? { OPERATOR_ACTION_KIND: params.context.actionKind } : {}),
        ...(params.context.command ? { OPERATOR_COMMAND: params.context.command } : {}),
        ...(params.context.title ? { OPERATOR_TITLE: params.context.title } : {}),
        ...(params.context.input ? { OPERATOR_HOOK_INPUT: JSON.stringify(params.context.input) } : {}),
      },
      maxBuffer: 1024 * 1024,
    });
    results.push({ event: params.context.event, command: hookCommand, stdout: stdout.trim(), stderr: stderr.trim() });
  }
  return results;
}

export async function runOperatorCommandHooks(params: {
  config: OperatorCliExecutionConfig;
  event: "PreCommand" | "PostCommand";
  action: OperatorExecutionAction;
}): Promise<OperatorHookResult[]> {
  return await runOperatorHooks({
    config: params.config,
    context: {
      event: params.event,
      cwd: params.action.cwd,
      permissionMode: params.config.permissionMode,
      ...(params.action.sessionId ? { sessionId: params.action.sessionId } : {}),
      actionKind: params.action.kind,
      command: params.action.command,
      ...(params.action.title ? { title: params.action.title } : {}),
      input: {
        source: params.action.source,
        action: params.action,
      },
    },
  });
}

function detectDangerousCommandReasons(command: string): string[] {
  const reasons: string[] = [];
  const checks: Array<{ regex: RegExp; reason: string }> = [
    { regex: /(^|\s)rm(\s|$)/, reason: "contains rm" },
    { regex: /(^|\s)dd(\s|$)/, reason: "contains dd" },
    { regex: /(^|\s)chmod(\s|$)/, reason: "contains chmod" },
    { regex: /(^|\s)chown(\s|$)/, reason: "contains chown" },
    { regex: /git\s+reset\b/, reason: "contains git reset" },
    { regex: /git\s+clean\b/, reason: "contains git clean" },
    { regex: /git\s+checkout\s+--\b/, reason: "contains git checkout --" },
    { regex: /git\s+restore\b/, reason: "contains git restore" },
    { regex: /git\s+push\s+--force\b/, reason: "contains git push --force" },
    { regex: /(^|\s)sudo(\s|$)/, reason: "contains sudo" },
    { regex: /(^|\s)kill(\s|$)/, reason: "contains kill" },
    { regex: /(^|\s)pkill(\s|$)/, reason: "contains pkill" },
    { regex: /(curl|wget)[^|\n]*\|\s*(bash|sh)\b/, reason: "contains fetch-and-exec pipeline" },
    { regex: /(^|\s)>>?(\s|$)/, reason: "contains shell redirection" },
  ];
  for (const check of checks) {
    if (check.regex.test(command)) {
      reasons.push(check.reason);
    }
  }
  return reasons;
}
