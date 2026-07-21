import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
    environment: "node",
    // Several integration suites (background tasks, operator runtime, control
    // plane) spawn real detached OS processes and make timing-sensitive
    // assertions on their state files. Running test files concurrently lets
    // those processes race across workers, producing non-deterministic
    // failures. Execute files serially so `npm test` is reproducible in the
    // cloud and on-device alike; individual suites remain fast.
    fileParallelism: false,
  },
});
