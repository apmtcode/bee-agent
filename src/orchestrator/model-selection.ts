import type { OperatorModelSelectionSource, OperatorResolvedModelSelection } from "../harness/types.js";

function normalizePrimary(value: string | undefined): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

function normalizeFallbacks(value: string[] | undefined): string[] | undefined {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : undefined;
}

export function buildResolvedModelSelection(params: {
  primary?: string;
  fallbacks?: string[];
  source: OperatorModelSelectionSource;
  preserveEmptyFallbacks?: boolean;
}): OperatorResolvedModelSelection | undefined {
  const primary = normalizePrimary(params.primary);
  const fallbacks = normalizeFallbacks(params.fallbacks);
  if (!primary && !params.preserveEmptyFallbacks && !fallbacks) {
    return undefined;
  }
  return {
    ...(primary ? { primary } : {}),
    ...(params.preserveEmptyFallbacks ? { fallbacks: fallbacks ?? [] } : fallbacks ? { fallbacks } : {}),
    source: params.source,
  };
}

export function cloneResolvedModelSelection(
  selection: OperatorResolvedModelSelection | undefined,
): OperatorResolvedModelSelection | undefined {
  if (!selection) {
    return undefined;
  }
  return buildResolvedModelSelection({
    primary: typeof selection.primary === "string" ? selection.primary : undefined,
    fallbacks: Array.isArray(selection.fallbacks)
      ? selection.fallbacks.filter((item): item is string => typeof item === "string")
      : undefined,
    source: selection.source,
    preserveEmptyFallbacks: Array.isArray(selection.fallbacks),
  });
}

export function resolvePatchedModelSelection(params: {
  baseSelection?: OperatorResolvedModelSelection;
  source: OperatorModelSelectionSource;
  primary?: string;
  hasPrimary?: boolean;
  fallbacks?: string[];
  hasFallbacks?: boolean;
}): OperatorResolvedModelSelection | undefined {
  if (!params.hasPrimary && !params.hasFallbacks) {
    return cloneResolvedModelSelection(params.baseSelection);
  }
  const primary = params.hasPrimary
    ? params.primary
    : params.baseSelection?.primary;
  const fallbacks = params.hasFallbacks
    ? Array.isArray(params.fallbacks)
      ? params.fallbacks.filter((item): item is string => typeof item === "string")
      : []
    : params.baseSelection?.fallbacks;
  return buildResolvedModelSelection({
    primary,
    fallbacks,
    source: params.source,
    preserveEmptyFallbacks: Boolean(
      params.hasFallbacks
      || (params.hasPrimary && !Array.isArray(params.baseSelection?.fallbacks)),
    ),
  });
}
