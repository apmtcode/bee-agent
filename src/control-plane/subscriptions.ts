import type { OperatorEvent, OperatorEventFilter } from "../kernel/event-bus.js";
import type { StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";

export type RuntimeEventSubscriptionOptions = {
  replay?: boolean;
  filter?: OperatorEventFilter<OperatorEvent>;
};

export function subscribeRuntimeEvents(
  runtime: StandaloneOperatorRuntime,
  options: RuntimeEventSubscriptionOptions = {},
): AsyncIterable<OperatorEvent> {
  return runtime.events.stream(options.filter, { replay: options.replay });
}
