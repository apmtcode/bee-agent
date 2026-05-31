export type SessionStatus = "active" | "idle" | "completed" | "failed";

export type OperatorModelSelectionSource = "job" | "inherited" | "override" | "default";

export type OperatorResolvedModelSelection = {
  primary?: string;
  fallbacks?: string[];
  source: OperatorModelSelectionSource;
};

export type SessionMetadata = {
  cwd?: string;
  title?: string;
  agentId?: string;
  remoteId?: string;
  remoteSource?: string;
  parentSessionId?: string;
  tags?: string[];
  modelSelection?: OperatorResolvedModelSelection;
};

export type SessionRecord = {
  id: string;
  createdAt: string;
  updatedAt: string;
  status: SessionStatus;
  transcriptFile?: string;
  metadata: SessionMetadata;
};

export type TranscriptMessageRole = "system" | "user" | "assistant" | "tool";
