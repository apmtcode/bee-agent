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
- `/pairing [status]`
- `/pairing create [source]`
- `/pairing approve <code|id>`
- `/pairing reject <code|id>`
- `/remote list [source]`
- `/remote status <remoteId|sessionId>`
- `/remote pause <remoteId|sessionId> [reason]`
- `/remote resume <remoteId|sessionId>`
- `/remote repair <remoteId|sessionId> [missing-process|missing-state]`
- `/platform list`
- `/platform status <platform>`
- `/platform pause <platform> [reason]`
- `/platform resume <platform>`
- `/recall <query>`
- `/tasks [team <name>]`
- `/task-create <subject> [--team <name>] [--owner <owner>] [--blocked-by <taskId,...>]`
- `/task-update <taskId> [<pending|in_progress|completed>] [--owner <owner|off>] [--blocked-by <taskId,...>]`
- `/messages`
- `/inbox`
- `/outbox`
- `/send <sessionId> <message>`
- `/notify <status> <message>`
- `/plan`
- `/plan-exit [default|acceptEdits|bypassPermissions]`
- `/statusline [command|off]`
- `/worktree [name]`
- `/worktree-exit [keep|remove]`
- `/teams`
- `/team-create <name> [description]`
- `/team-delete <name>`
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

## Task lists

Operator now also has a first narrow persisted task-list surface for multi-step work tracking.

Current task behavior in this tranche:
- tasks are persisted as session-scoped records with `pending`, `in_progress`, and `completed` status
- task records keep structured fields for `description`, optional `activeForm`, optional `owner`, `metadata`, `blocks`, and `blockedBy`
- the runtime exposes create, list, get, and update flows through the existing control-plane RPC surface as `tasks.create`, `tasks.list`, `tasks.get`, and `tasks.update`
- task lifecycle changes emit `task.created` and `task.updated` runtime events and can be replayed or filtered through the `task` event family
- bootstrapped session streams inherit `sessionId` for `tasks.create` and `tasks.list`
- the CLI exposes `/tasks [team <name>]`, `/task-create <subject> [--team <name>] [--owner <owner>] [--blocked-by <taskId,...>]`, and `/task-update <taskId> [<pending|in_progress|completed>] [--owner <owner|off>] [--blocked-by <taskId,...>]` for narrow local task inspection and coordination updates
- team ownership rules, assignment arbitration, and richer dependency editing syntax remain out of scope in this tranche

## Session messaging

Operator now also has a first narrow persisted mailbox surface for session-to-session coordination.

Current messaging behavior in this tranche:
- messages are persisted as append-only records with `fromSessionId`, `toSessionId`, `summary`, `message`, optional `metadata`, and timestamps
- the runtime exposes send, get, list, inbox, and outbox flows through the existing control-plane RPC surface as `messages.send`, `messages.get`, `messages.list`, `messages.inbox`, and `messages.outbox`
- sending a message emits a `message.sent` runtime event and can be replayed or filtered through the `message` event family
- bootstrapped session streams inherit `fromSessionId` for `messages.send` and `sessionId` for `messages.list`, `messages.inbox`, and `messages.outbox`
- the CLI exposes `/messages`, `/inbox`, `/outbox`, and `/send <sessionId> <message>` for narrow local mailbox inspection and delivery
- richer mailbox threading, unread state, assignment semantics, and team arbitration remain out of scope in this tranche

## Plan mode

Operator now also has a first narrow Claude Code-style plan mode surface for local planning-only sessions.

Current plan mode behavior in this tranche:
- the CLI exposes `/plan` to persist `permissionMode: "plan"` in `<cwd>/.claude/settings.local.json`
- entering plan mode remembers the prior non-plan local permission mode under `operator.previousPermissionMode` when one exists
- `/plan-exit [default|acceptEdits|bypassPermissions]` restores the remembered mode or a caller-specified override
- command execution continues to honor the existing execution-policy rule that blocks execution while `permissionMode=plan`
- this tranche remains intentionally narrow and does not yet add explicit plan documents, approval handoff, or a separate enter/exit plan runtime protocol

## Status line setup

Operator now also has a first narrow status-line configuration surface for Claude Code-style local session visibility.

Current status-line behavior in this tranche:
- the CLI exposes `/statusline [command|off]` to inspect, set, or remove a local status-line command
- status-line config is stored in `<cwd>/.claude/settings.local.json` under `statusLine`
- the supported shape mirrors the narrow Claude Code command-backed form: `{ "type": "command", "command": "..." }`
- exported helpers can resolve the merged `statusLine` command config or persist a project-local override
- this tranche remains intentionally narrow and does not yet execute the command, stream session JSON to it, or render a live terminal status line

## Worktree entry and exit

Operator now also has a first narrow worktree surface for Claude Code-style isolated local edits.

Current worktree behavior in this tranche:
- the CLI exposes `/worktree [name]` to create and enter a git worktree rooted under `<repo>/.operator/worktrees/`
- each created worktree gets an isolated branch named `operator/<name>`
- `/worktree-exit [keep|remove]` returns the CLI session to the original cwd and can either keep the worktree or remove both the worktree and its branch
- this tranche is intentionally CLI-local and git-only; it does not yet add session handoff, existing-worktree attach, dirty-state protection, or non-git hook-backed worktrees

## Team registry

Operator now also has a first narrow team surface for Claude Code-style multi-agent coordination setup.

Current team behavior in this tranche:
- the CLI exposes `/teams`, `/team-create <name> [description]`, and `/team-delete <name>`
- each created team now gets a dedicated persisted task session so the existing task store can coordinate shared team work instead of staying tied to the caller session
- team-scoped task flows reuse the existing task surface through `/tasks team <name>` and `--team <name>` task creation flags
- task coordination can now persist `owner` and `blockedBy` through the CLI for team task handoff and dependency tracking
- the public operator export surface now includes the team store types for later runtime or control-plane integration
- this tranche remains intentionally narrow and does not yet add teammate spawning, routing, or shutdown orchestration

## Push notifications

Operator now also has a first narrow push-notification surface for local operator attention routing.

Current notification behavior in this tranche:
- notifications are delivered through the existing delivery seam instead of adding a separate transport path
- the runtime exposes a narrow `sendNotification()` flow and emits `notification.sent` runtime events
- the control plane exposes `notifications.send` with optional explicit delivery targets and local delivery by default
- runtime event replay/filtering now includes the `notification` family alongside existing run, task, message, and subagent families
- bootstrapped session streams inherit `sessionId` for `notifications.send`
- the CLI exposes `/notify <status> <message>` for first-class local push notification delivery
- this tranche remains intentionally narrow and does not yet add notification history, mobile platform integrations, retry queues, or acknowledgement state

## Remote session stream

The control plane now also exposes a first remote session-stream seam for future gateway/channel work.

Current remote session behavior in this tranche:
- `sessions.bootstrap` can create or reattach to a session and optionally resume idle sessions
- bootstrap returns session-scoped pending approvals, a replayable task-plan snapshot, plus a filtered replay of runtime events
- `OperatorControlPlaneSessionStream` binds later session-scoped control-plane requests to the bootstrapped session
- the same stream adapter can subscribe to live session-scoped runtime events for approvals, runs, tasks, skills, and background tasks
- the seam is transport-agnostic and in-process in this tranche; it is intended to be mounted by a later gateway/channel transport layer

## Gateway transport

Operator now also has a first minimal gateway transport layer on top of that session-stream seam.

Current gateway transport behavior in this tranche:
- one remote connection can bootstrap or reattach to a single operator session
- later connection-scoped requests are forwarded through the bound session stream without resupplying `sessionId`
- live session-scoped runtime events are forwarded as transport event frames
- transport bootstrap now includes the session task-plan snapshot so reconnecting clients can recover persisted operator tasks immediately
- transport message types remain intentionally narrow and now include heartbeat `ping` / `pong` alongside bootstrap, request, response, event, and error
- gateway bootstrap can now also carry a stable `remoteId` and `remoteSource` so reconnecting clients can reattach without already knowing `sessionId`
- reconnecting clients can optionally provide an event replay cursor so bootstrap only replays missed session-scoped events
- stale gateway heartbeats now quarantine the affected remote until an operator explicitly resumes it, instead of silently dropping continuity or auto-clearing on a later healthy heartbeat
- bootstrap can also redeem an approved `pairingCode` into that same `remoteId` continuity path
- this is still an in-process transport adapter in this tranche; broader channel delivery, platform routing, and circuit-breaker policy remain for later work

## Pairing bootstrap

Operator now has a first user-facing pairing/bootstrap seam on top of remote session continuity.

Current pairing behavior in this tranche:
- the control plane can create durable short-lived pairing tickets with a human-usable code plus `remoteId` and `remoteSource`
- pairing tickets move through explicit states: `pending`, `approved`, `rejected`, `redeemed`, and `expired`
- the operator CLI can list tickets and approve or reject pending claims with `/pairing`
- `sessions.bootstrap` can redeem an approved `pairingCode` and continue through the existing session bootstrap path
- redeemed codes are single-use and later reuse is rejected

## Remote status

Operator now also has a first narrow remote diagnostics surface on top of pairing and remote session continuity.

Current remote status behavior in this tranche:
- the control plane exposes `sessions.remoteInventory` for fleet-style remote summaries and `sessions.remoteStatus` for per-remote drill-down
- remote inventory summarizes known remotes from active/idle sessions plus pending or approved-unredeemed pairing claims, with optional `remoteSource` filtering
- remote status resolves the bound local session plus `remoteSource`
- remote status includes the latest pairing ticket for that remote when one exists
- remote status reports session-scoped pending approvals, the active run, the active background task, and recent runtime events
- remote status and inventory now project a derived remote `control` state of `active`, `paused`, `quarantined`, or `degraded`
- quarantined remotes also surface a recoverable diagnostics block with a recommended `/remote resume <remoteId|sessionId>` action
- the operator CLI exposes fleet summaries with `/remote list [source]` and drill-down troubleshooting with `/remote status <remoteId|sessionId>`

## Platform fleet view

Operator now also has a first grouped platform surface on top of remote status and control.

Current platform behavior in this tranche:
- the control plane exposes `sessions.platformInventory`, `sessions.platformStatus`, and `sessions.platformControl`
- platforms are currently derived from existing remote `remoteSource` values rather than a new durable routing model
- platform inventory summarizes grouped remote count plus an aggregate `control` state
- platform status returns the member remotes for a platform and can surface a grouped `mixed` state when members differ
- platform pause and resume persist a platform-level breaker keyed by normalized platform name, so later inventory/status calls retain the platform-level control summary
- repeated retryable remote failures now accumulate as automatic platform observations and escalate a platform from `degraded` to `paused`
- automatic breaker summaries now expose `source`, `failureCount`, and `threshold` metadata in the control-plane payload so later replay/UI work can distinguish manual vs automatic pauses
- when an automatic breaker reaches `paused`, active runs on that platform are paused with `remoteControlSource="platform-breaker"`, while explicit `/platform pause` and `/remote pause` continue to use the existing manual control path
- per-remote control and diagnostics remain authoritative for individual remotes; manual pause/resume still overrides and clears automatic observations for that platform
- the operator CLI exposes `/platform list`, `/platform status <platform>`, `/platform pause <platform> [reason]`, and `/platform resume <platform>`
- platform routing, account selection, richer retry classification, and durable platform identity remain out of scope in this tranche

## Remote subagent spawn

Operator now also has a first narrow remote child-work orchestration seam on top of the existing session stream and gateway transport.

Current remote subagent behavior in this tranche:
- the control plane exposes `subagents.spawn` to create a child session, register a subagent against a parent run, and start the child run in one request
- bootstrapped session streams inherit the parent `sessionId` for `subagents.spawn`, so remotes can branch work without resupplying local session identity
- runtime event subscriptions now support the `subagent` family for replay/live inspection of `subagent.registered` and `subagent.updated`
- gateway transport forwards the new RPC and subagent event family through the existing bootstrap/request/event frames
- richer cron fallback policy, account routing, and transport-specific child-channel semantics remain out of scope in this tranche

## Remote pause, degraded control, and repair

Operator now also has a first narrow remote-control seam on top of remote status diagnostics.

Current remote control behavior in this tranche:
- the control plane exposes `sessions.remoteControl` for `pause` and `resume` against a resolved `remoteId` or `sessionId`
- pausing acts on the active top-level run for that remote session and records remote-control metadata on the run
- stale gateway heartbeat failures trip an in-memory quarantine registry keyed by `remoteId`
- `/remote pause <remoteId|sessionId> [reason]` lets the operator stop delegated progress and immediately inspect the updated state
- `/remote resume <remoteId|sessionId>` clears quarantine for that remote and resumes the bound session when applicable
- `sessions.remoteStatus` and `/remote status` now surface a derived `control` summary with `active`, `paused`, `quarantined`, or `degraded`
- quarantined heartbeat failures recommend an explicit `/remote resume <remoteId|sessionId>` action and do not auto-clear on a later healthy heartbeat
- degraded status can still include a small diagnostics block with the current cause, whether it is recoverable, the relevant task when known, and a recommended repair command
- the control plane also exposes `sessions.remoteRepair` for scoped `recover-background-tasks` against the resolved remote session
- `/remote repair <remoteId|sessionId> [missing-process|missing-state]` reuses existing background-task recovery and reports repaired tasks plus refreshed control state
- this repair path is intentionally limited to background-task drift recovery; process suspension, platform-specific routing, and broader circuit-breaker policy remain out of scope in this tranche

## Settings precedence

The CLI loads settings in this order, with later files overriding earlier ones:
1. `~/.claude/settings.json` or `$CLAUDE_CONFIG_HOME/settings.json`
2. `<cwd>/.claude/settings.json`
3. `<cwd>/.claude/settings.local.json`

Nested objects are deep-merged so project/local overrides can extend shared user config instead of replacing it entirely.

## Execution policy

The local CLI now interprets a narrow execution policy from merged settings:
- `permissionMode`: `default`, `acceptEdits`, `bypassPermissions`, or `plan`
- command hooks: first-class `hooks.PreToolUse`, plus legacy `hooks.PreCommand`, and `hooks.PostCommand`
- lifecycle hooks: `hooks.SessionStart`, `hooks.SessionEnd`, `hooks.ApprovalRequested`, and `hooks.ApprovalResolved`
- `hooks.PostToolUse` currently remains an alias to `hooks.PostCommand`

Covered command execution in this tranche:
- `/background start <title> -- <command>`
- executable skill `command` steps

Covered lifecycle execution in this tranche:
- session creation via `startSession()`
- session terminal transitions via `completeSession()` and `failSession()`
- approval creation via `promptApproval()`
- approval resolution via `resolveApproval()`

In `default` and `acceptEdits` modes, dangerous commands require approval before they run. The initial dangerous-command heuristic covers destructive filesystem commands, risky git mutation, privilege-escalation commands, fetch-and-exec pipelines, and shell redirection. Use `/approvals`, `/approve <id>`, and `/deny <id>` to manage pending approvals.

Configured hooks run synchronously with these shared environment variables when relevant:
- `OPERATOR_HOOK_EVENT`
- `OPERATOR_SESSION_ID`
- `OPERATOR_RUN_ID`
- `OPERATOR_APPROVAL_ID`
- `OPERATOR_CWD`
- `OPERATOR_PERMISSION_MODE`
- `OPERATOR_HOOK_INPUT`

Command hooks also receive these command-specific environment variables:
- `OPERATOR_ACTION_KIND`
- `OPERATOR_COMMAND`
- `OPERATOR_TITLE`

This remains a narrow synchronous hook pass. `PreToolUse` can synchronously allow, deny, request approval, and minimally rewrite `command` / `title` for covered command-like execution. Async hook protocols, `PostToolUse` result mutation, failure hooks, matcher-targeted hooks, file or cwd watch hooks, and broader Claude-style tool-surface parity remain out of scope in this tranche.

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
- cron jobs can now optionally declare model primary/fallback metadata
- cron-dispatched runs can inherit a base model-selection context when the job omits overrides
- job-level cron overrides can replace primary and fallback lists independently, including fallback-only overrides
- cron runs and spawned subagents now share the same model-selection patching semantics, including strict primary-only overrides and explicit empty fallback overrides
- explicit empty fallback lists are preserved instead of being silently re-inherited
- cron-dispatched sessions, top-level operator runs, and persisted cron runs now carry the resolved model-selection context they used
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

## Subagent model inheritance

Operator now also preserves model-selection context when runtime work branches into child sessions.

Current subagent model behavior in this tranche:
- `spawnSubagent()` now inherits model-selection context from the parent run first, then the parent session when no explicit child override is provided
- child sessions and child runs persist the resolved model primary/fallback selection as metadata for later inspection and future routing work
- explicit child overrides can replace the primary model while still inheriting fallbacks when unspecified
- explicit empty fallback lists are preserved as a real override instead of being silently re-inherited
- this remains metadata-only in this tranche; provider routing, retry/fallback execution, and a user-facing `/model` surface remain out of scope

## Anthropic HTTP compatibility

Operator now also has a first narrow Anthropic-compatible HTTP seam alongside the RPC control plane.

Current behavior in this tranche:
- the control plane can now forward both non-streaming and SSE streaming `POST /v1/messages` requests through a minimal Anthropic client path
- forwarded requests send Claude-style `x-api-key` auth and optional `Authorization: Bearer ...` when `ANTHROPIC_AUTH_TOKEN` is present
- base URL resolution now honors `ANTHROPIC_BASE_URL`
- incoming `anthropic-beta` headers are now forwarded through the control plane
- Claude-native request fields like `system`, `tool_choice`, `stop_sequences`, `temperature`, `top_p`, `top_k`, `metadata`, and `thinking` are preserved through request normalization instead of being dropped by the control-plane seam
- request bodies now perform narrow Anthropic-oriented sanitation for message histories by stripping orphaned tool blocks, merging same-role neighbors, and removing trailing assistant prefill when `thinking` is enabled unless the final assistant turn is a retained tool use
- non-streaming and streaming responses preserve key upstream transport metadata like `request-id`, Anthropic request IDs, selected rate-limit headers, and `retry-after`
- streaming responses now pass through raw `text/event-stream` bodies instead of being rejected at the control-plane boundary
- this tranche is intentionally narrow and still does not add retries, SSE event normalization, broader provider routing, or container/MCP-specific request shaping

## Current gaps to parity

The system is still incomplete relative to the long-term goal. Major missing tranches include:
- richer Claude Code style command surface and hook execution
- fuller OpenClaw-style channel/gateway runtime and remote transport beyond the current heartbeat/reconnect-aware gateway transport
- richer remote identity/account routing beyond the current first pairing-code bootstrap seam
- richer delivery routing, retries, and transport semantics beyond local/webhook terminal notifications
- richer tool-loop and broader command-surface parity in the CLI shell beyond the current local run/task watch surfaces

This README should be updated as each tranche lands so it remains the top-level project map.