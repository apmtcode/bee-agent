import type { OperatorRunRecord } from "./run-store.js";

export type ExecutionGraphNode = {
  run: OperatorRunRecord;
  children: ExecutionGraphNode[];
};

export function buildExecutionGraph(runs: OperatorRunRecord[]): ExecutionGraphNode[] {
  const nodes = new Map<string, ExecutionGraphNode>(runs.map((run) => [run.id, { run, children: [] }]));
  const roots: ExecutionGraphNode[] = [];
  for (const run of runs) {
    const node = nodes.get(run.id);
    if (!node) {
      continue;
    }
    if (run.parentRunId) {
      const parent = nodes.get(run.parentRunId);
      if (parent) {
        parent.children.push(node);
        continue;
      }
    }
    roots.push(node);
  }
  return roots;
}
