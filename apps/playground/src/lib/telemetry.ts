export interface RequestTelemetryV1 {
  schema_version: 1;
  request_id: string;
  model: string;
  admission?: { queue_wait_ms: number | null; outcome: string };
  execution?: { started_ms?: number | null; prefill_start_ms?: number | null; prefill_end_ms?: number | null; first_token_ms?: number | null; last_token_ms?: number | null; completed_ms?: number | null };
  counts?: { prompt_tokens?: number | null; generated_tokens?: number | null };
  tokens?: { ids?: number[]; latencies_ms?: Array<number | null> };
  runtime?: Record<string, unknown>;
  memory?: Record<string, unknown>;
  cache?: Record<string, unknown>;
  outcome?: string;
  reason?: string | null;
}
export function parseTelemetry(value: unknown): RequestTelemetryV1 | null {
  if (!value || typeof value !== 'object') return null;
  const v = value as Record<string, unknown>;
  if (v.schema_version !== 1 || typeof v.request_id !== 'string' || typeof v.model !== 'string') return null;
  return v as unknown as RequestTelemetryV1;
}
