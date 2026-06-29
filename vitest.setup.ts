// Global test setup: run background-task launches in inert (dry-run) mode so the
// suite never spawns real detached OS processes that would race with tests that
// drive task execution state directly. See defaultSpawnBackgroundProcess in
// src/harness/background-tasks.ts.
process.env.OPENCLAW_BACKGROUND_SPAWN = "inert";
