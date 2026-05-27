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
- `/watch run [runId]`
- `/watch task [taskId]`
- `/watch active`
- `/training`
- `/cron`
- `/cron create <cronExpr> <prompt>`
- `/cron runs [jobId]`
- `/cron tick`
- `/cron delete <jobId>`
- `/config`
- `/prompt`

Freeform conversational turns now run through a narrow local assistant path in the CLI. Each turn is grounded by prompt-context discovery plus recent session transcript history, persisted through the existing runtime/session/transcript flow, and can reuse the current approval, executable-skill, background-task, recall, and inspection surfaces without dropping back to synthetic placeholder responses.

Local freeform turns now also emit live run-progress updates while they execute. The CLI surfaces in-flight progress for freeform handling, approval waits, skill execution, and background-task-backed actions before printing the final assistant response.

The local shell now also has a first watch/follow surface for long-lived work. `/watch run`, `/watch task`, and `/watch active` let the operator inspect the current state of active runs and background tasks without falling back to manual polling workflows.

## Remote session stream

The control plane now also exposes a first remote session-stream seam for future gateway/channel work.

Current remote session behavior in this tranche:
- `sessions.bootstrap` can create or reattach to a session and optionally resume idle sessions
- bootstrap returns session-scoped pending approvals plus a filtered replay of runtime events
- `OperatorControlPlaneSessionStream` binds later session-scoped control-plane requests to the bootstrapped session
- the same stream adapter can subscribe to live session-scoped runtime events for approvals, runs, skills, and background tasks
- the seam is transport-agnostic and in-process in this tranche; it is intended to be mounted by a later gateway/channel transport layer

## Gateway transport

Operator now also has a first minimal gateway transport layer on top of that session-stream seam.

Current gateway transport behavior in this tranche:
- one remote connection can bootstrap or reattach to a single operator session
- later connection-scoped requests are forwarded through the bound session stream without resupplying `sessionId`
- live session-scoped runtime events are forwarded as transport event frames
- transport message types are intentionally narrow: bootstrap, request, response, event, and error
- gateway bootstrap can now also carry a stable `remoteId` and `remoteSource` so reconnecting clients can reattach without already knowing `sessionId`
- this is still an in-process transport adapter in this tranche; user-facing pairing/bootstrap UX and broader channel delivery remain for later work

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
- cron jobs can optionally declare terminal-state delivery targets for success and failure paths
- per-run delivery attempts are persisted independently from execution outcome

Current delivery behavior in this tranche:
- supported delivery targets are `local` and `webhook`
- delivery runs only after a cron run reaches terminal state
- delivery failure does not rewrite a successful cron execution into a failed execution
- local delivery writes a durable local notification artifact
- webhook delivery posts a small JSON envelope with no retries in this tranche

Current CLI visibility for this tranche:
- `/cron` now shows `next`, `last`, `status`, delivery summary, and any last execution or delivery error
- `/cron create <cronExpr> <prompt> [--on-success <target>] [--on-failure <target>]` can attach terminal delivery targets
- `/cron runs [jobId]` and `/cron tick` show linked session/run ids plus terminal outcome and delivery details

This is still a narrow outbound-only pass. Cron expression handling remains placeholder, and operator still does not have interactive channel/gateway runtime parity.

## Current gaps to parity

The system is still incomplete relative to the long-term goal. Major missing tranches include:
- richer Claude Code style command surface and hook execution
- fuller OpenClaw-style channel/gateway runtime and remote transport beyond the current minimal gateway transport
- user-facing pairing/bootstrap UX and richer remote identity/account routing on top of the current remoteId continuity layer
- richer delivery routing, retries, and transport semantics beyond local/webhook terminal notifications
- richer tool-loop and broader command-surface parity in the CLI shell beyond the current local run/task watch surfaces

This README should be updated as each tranche lands so it remains the top-level project map.