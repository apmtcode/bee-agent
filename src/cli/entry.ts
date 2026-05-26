#!/usr/bin/env node
import process from "node:process";
import { OperatorCliApp } from "./app.js";

async function main() {
  const rootDir = process.argv[2] || process.cwd();
  const app = new OperatorCliApp({ rootDir, cwd: process.cwd() });
  await app.runRepl();
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exit(1);
});
