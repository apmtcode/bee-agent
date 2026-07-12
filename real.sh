#!/usr/bin/env bash
set -euo pipefail
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s' '{"version":1,"taskId":"3a008657-fb68-4fbc-8d15-ed7e6c73d354","kind":"monitor","status":"running","pid":"$$","startedAt":"__OPENCLAW_STARTED_AT__","updatedAt":"__OPENCLAW_STARTED_AT__","outputFile":"tasks/x/output.log","cwd":"/tmp/foo","command":"tail -f app.log"}' | sed "s/__OPENCLAW_STARTED_AT__/$started_at/g; s/"$$"/$$/g" > state2.json
