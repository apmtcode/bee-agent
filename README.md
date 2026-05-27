# Standalone Operator

`operator/` is the standalone assistant runtime being built inside this workspace. It is intended to converge toward the combined capabilities of OpenClaw, Hermes Agent, and Claude Code / Claw Code while staying structurally independent.

## What it does today

The current standalone operator already includes:
- session, transcript, approval, run, and subagent orchestration
- reviewed user-behavior capture and trajectory storage
- replay manifests and replay-backed recall
- memory items, promoted skills, and executable skills with usage/provenance
- plugin registration and runtime activation
- reviewed export generation for local training
- local Apple Silicon training job planning and execution tracking
- a minimal CLI shell with layered settings, scoped prompt loading, and slash commands

## Architecture

The project is organized around a few core layers:
- `src/orchestrator/` — primary runtime entrypoint and orchestration state
- `src/control-plane/` — RPC surface for sessions, runs, skills, capture, training, plugins, and cron
- `src/capture/` — consent-aware behavior capture, trajectory storage, and replay generation
- `src/memory/` — memories, promoted skills, executable skills, and recall types/store
- `src/training/` — reviewed export, local training manifests, runner plans, and execution state
- `src/plugins/` — manifest registry and plugin runtime activation
- `src/cli/` — layered settings loading, prompt-context discovery, and the standalone operator shell

## CLI

The package now exposes an `operator` CLI via `dist/cli/entry.mjs`.

Current slash commands:
- `/help`
- `/status`
- `/sessions`
- `/resume <sessionId>`
- `/idle`
- `/complete`
- `/transcript [limit]`
- `/approvals`
- `/approve <approvalId>`
- `/deny <approvalId>`
- `/recall <query>`
- `/skills`
- `/run-skill <id>`
- `/background`
- `/background start <title> -- <command>`
- `/background view <taskId> [lines]`
- `/background sync <taskId>`
- `/background cancel <taskId>`
- `/training`
- `/cron`
- `/cron create <cronExpr> <prompt>`
- `/cron runs [jobId]`
- `/cron tick`
- `/cron delete <jobId>`
- `/config`
- `/prompt`

Freeform conversational turns are not implemented yet, but the CLI now acts as a stateful local control shell over the existing runtime for session, approval, background-task, cron, and prompt/config inspection workflows.

## Settings precedence

The CLI loads settings in this order, with later files overriding earlier ones:
1. `~/.claude/settings.json` or `$CLAUDE_CONFIG_HOME/settings.json`
2. `<cwd>/.claude/settings.json`
3. `<cwd>/.claude/settings.local.json`

Nested objects are deep-merged so project/local overrides can extend shared user config instead of replacing it entirely.

## Execution policy

The local CLI now interprets a narrow execution policy from merged settings:
- `permissionMode`: `default`, `acceptEdits`, `bypassPermissions`, or `plan`
- `hooks.PreCommand` / `hooks.PostCommand`
- `hooks.PreToolUse` / `hooks.PostToolUse` as aliases for local command actions

Covered command execution in this tranche:
- `/background start <title> -- <command>`
- executable skill `command` steps

In `default` and `acceptEdits` modes, dangerous commands require approval before they run. The initial dangerous-command heuristic covers destructive filesystem commands, risky git mutation, privilege-escalation commands, fetch-and-exec pipelines, and shell redirection. Use `/approvals`, `/approve <id>`, and `/deny <id>` to manage pending approvals.

Configured pre/post hooks run synchronously around covered command execution with these environment variables:
- `OPERATOR_HOOK_EVENT`
- `OPERATOR_ACTION_KIND`
- `OPERATOR_SESSION_ID`
- `OPERATOR_CWD`
- `OPERATOR_COMMAND`
- `OPERATOR_PERMISSION_MODE`

## Prompt and instruction discovery

The CLI walks ancestor directories from the filesystem root to the working directory and loads instruction files in stable order from these locations when present:
- `CLAUDE.md`
- `CLAUDE.local.md`
- `.claude/CLAUDE.md`

The prompt context also records the working directory, current date, and a lightweight git-status snapshot when available.

## Verification

Focused tests for this package should use the local wrapper from the repo root:

`node openclaw/scripts/run-vitest.mjs <explicit operator test paths>`

When public/runtime/package surfaces change, rebuild with:

`openclaw/node_modules/.bin/tsdown operator/src/index.ts operator/src/cli/entry.ts --tsconfig operator/tsconfig.json --no-config --platform node --format esm --dts --out-dir operator/dist --clean`

## Cron dispatch

Cron jobs now dispatch into real operator runtime artifacts instead of completing as synthetic bookkeeping rows.

Current cron behavior in this tranche:
- each due cron fire creates a fresh session and runtime run
- cron runs record linked `sessionId` and `operatorRunId`
- successful cron fires record a real transcript turn and complete the linked session/run
- failed cron fires persist `lastStatus` / `lastError` on the job and fail the linked session/run
- stale `scheduled` / `running` cron runs are recovered as interrupted failures on the next tick

Current CLI visibility for this tranche:
- `/cron` now shows `next`, `last`, `status`, and any last error
- `/cron runs [jobId]` and `/cron tick` show linked session/run ids plus terminal outcome details

This is still a local first pass. Cron expression handling remains placeholder, and cron does not yet dispatch into channel/gateway transports.

## Current gaps to parity

The system is still incomplete relative to the long-term goal. Major missing tranches include:
- richer Claude Code style command surface and hook execution
- OpenClaw-style channel/gateway runtime and remote transport
- more complete cron scheduling, delivery, and transport integration
- more complete interactive assistant behavior in the CLI shell

This README should be updated as each tranche lands so it remains the top-level project map.