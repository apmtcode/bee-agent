import { execFile } from "node:child_process";
import process from "node:process";
import { promisify } from "node:util";
import type { OperatorCliExecutionConfig, OperatorCliPermissionMode } from "./config.js";

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
  event: "PreCommand" | "PostCommand";
  command: string;
  stdout: string;
  stderr: string;
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

export async function runOperatorCommandHooks(params: {
  config: OperatorCliExecutionConfig;
  event: "PreCommand" | "PostCommand";
  action: OperatorExecutionAction;
}): Promise<OperatorHookResult[]> {
  const commands = params.config.hooks[params.event];
  const results: OperatorHookResult[] = [];
  for (const hookCommand of commands) {
    const { stdout, stderr } = await execFileAsync("bash", ["-lc", hookCommand], {
      cwd: params.action.cwd,
      env: {
        ...process.env,
        OPERATOR_HOOK_EVENT: params.event,
        OPERATOR_ACTION_KIND: params.action.kind,
        ...(params.action.sessionId ? { OPERATOR_SESSION_ID: params.action.sessionId } : {}),
        OPERATOR_CWD: params.action.cwd,
        OPERATOR_COMMAND: params.action.command,
        OPERATOR_PERMISSION_MODE: params.config.permissionMode,
      },
      maxBuffer: 1024 * 1024,
    });
    results.push({ event: params.event, command: hookCommand, stdout: stdout.trim(), stderr: stderr.trim() });
  }
  return results;
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
