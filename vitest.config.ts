import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
    environment: "node",
    // Background-task launches spawn real detached processes that write state
    // files asynchronously; in tests that race with explicit writeState/
    // writeOutput calls. Dry-launch keeps the suite deterministic. Individual
    // tests that need the real launcher clear this var locally.
    env: {
      OPENCLAW_BACKGROUND_TASK_DRY_RUN: "1",
    },
  },
});
