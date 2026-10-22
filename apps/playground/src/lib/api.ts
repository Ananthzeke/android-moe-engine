export type Role = 'system' | 'user' | 'assistant';
export interface ChatMessage { role: Role; content: string; status?: string; usage?: Usage; metrics?: GenerationMetrics; }
export interface Usage { prompt_tokens: number; completion_tokens: number; total_tokens: number; }
export interface Health { status: string; model: string; queue?: { outstanding: number; active: boolean; admitted: number; rejected: number; queue_timeouts: number }; streaming?: Record<string, number | boolean>; error?: string; }
import { parseTelemetry, type RequestTelemetryV1 } from './telemetry';
export interface GenerationMetrics {
  source: 'browser+backend-usage';
  ttft_ms: number | null;
  total_ms: number;
  decode_tps: number | null;
  completion_tokens: number | null;
  token_latency_reason: string | null;
  backend_telemetry: RequestTelemetryV1 | null;
}
export interface ChatOptions { model: string; temperature: number; max_tokens: number; seed: number; stream: boolean; stream_options?: { include_usage: boolean }; }
export interface StreamResult { content: string; usage?: Usage; metrics: GenerationMetrics; finishReason: string | null; }

const endpoint = '/api/backend';

export async function getHealth(): Promise<Health> {
  const r = await fetch(endpoint + '/health', { headers: { accept: 'application/json' } });
  if (!r.ok) { let body: unknown = null; try { body = await r.json(); } catch {} throw new Error(errorMessage(r.status, body)); }
  const value: unknown = await r.json();
  if (!value || typeof value !== 'object' || !('status' in value) || typeof value.status !== 'string' || !('model' in value) || typeof value.model !== 'string') throw new Error('Phone health response malformed.');
  return value as Health;
}
export async function getModels(): Promise<string[]> {
  const r = await fetch(endpoint + '/v1/models');
  if (!r.ok) { let body: unknown = null; try { body = await r.json(); } catch {} throw new Error(errorMessage(r.status, body)); }
  const body = await r.json();
  if (!Array.isArray(body.data)) throw new Error('model list malformed');
  return body.data.map((m: {id?: string}) => m.id).filter((x: unknown): x is string => typeof x === 'string');
}
function errorMessage(status: number, body: unknown): string {
  if (body && typeof body === 'object' && 'error' in body) {
    const e = (body as {error?: {message?: string; code?: string}}).error;
    if (e?.message) return e.message + (e.code ? ' (' + e.code + ')' : '');
  }
  if (status === 429) return 'Backend is at its four-request admission limit (HTTP 429).';
  if (status === 504) return 'The request waited 300 seconds for a generation slot (HTTP 504).';
  return 'Backend request failed: HTTP ' + status;
}
export async function streamChat(messages: ChatMessage[], options: ChatOptions, signal: AbortSignal, onEvent: (delta: string, metrics?: GenerationMetrics, phase?: string) => void): Promise<StreamResult> {
  const started = performance.now();
  let ttft: number | null = null;
  let firstAt: number | null = null;
  let lastAt: number | null = null;
  let content = '';
  let usage: Usage | undefined;
  let finishReason: string | null = null;
  let backendTelemetry: RequestTelemetryV1 | null = null;
  const r = await fetch(endpoint + '/v1/chat/completions', {
    method: 'POST', signal,
    headers: {'content-type': 'application/json', accept: 'text/event-stream'},
    body: JSON.stringify({ ...options, messages })
  });
  if (!r.ok) {
    let body: unknown = null; try { body = await r.json(); } catch {}
    throw new Error(errorMessage(r.status, body));
  }
  if (!r.body) throw new Error('backend returned no response body');
  onEvent('', undefined, 'waiting');
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      let split: number;
      while ((split = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, split); buffer = buffer.slice(split + 2);
        for (const line of block.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (!data || data === '[DONE]') continue;
          let event: any; try { event = JSON.parse(data); } catch { throw new Error('malformed SSE JSON'); }
          if (event.error) throw new Error(event.error.message || 'backend stream error');
          if (event.usage) usage = event.usage;
          if (event.ame_telemetry) backendTelemetry = parseTelemetry(event.ame_telemetry);
          const choice = event.choices?.[0];
          if (!choice) continue;
          if (choice.finish_reason) finishReason = choice.finish_reason;
          const delta = choice.delta?.content;
          if (typeof delta === 'string' && delta.length > 0) {
            const at = performance.now();
            if (firstAt === null) { firstAt = at; ttft = at - started; }
            lastAt = at; content += delta;
            onEvent(delta, metricsNow(started, ttft, firstAt, lastAt, usage?.completion_tokens ?? null, backendTelemetry), 'generating');
          }
        }
      }
    }
    buffer += decoder.decode();
  } finally { reader.releaseLock(); }
  if (!finishReason) throw new Error('Response stream ended before a completion event. Retry when the phone is ready.');
  const total = performance.now() - started;
  const n = usage?.completion_tokens ?? null;
  const decode = firstAt !== null && lastAt !== null && n !== null && n >= 2 && lastAt > firstAt ? (n - 1) / ((lastAt - firstAt) / 1000) : null;
  const reason = decode === null ? (n === null ? 'missing backend usage' : n < 2 ? 'fewer than two generated tokens' : 'non-positive token interval') : null;
  const metrics: GenerationMetrics = { source: 'browser+backend-usage', ttft_ms: ttft, total_ms: total, decode_tps: decode, completion_tokens: n, token_latency_reason: reason, backend_telemetry: backendTelemetry };
  return { content, usage, metrics, finishReason };
}
function metricsNow(started: number, ttft: number | null, first: number | null, last: number | null, n: number | null, telemetry: RequestTelemetryV1 | null): GenerationMetrics {
  const decode = first !== null && last !== null && n !== null && n >= 2 && last > first ? (n - 1) / ((last - first) / 1000) : null;
  const reason = decode === null ? (n === null ? 'missing backend usage' : n < 2 ? 'fewer than two generated tokens' : 'non-positive token interval') : null;
  return { source: 'browser+backend-usage', ttft_ms: ttft, total_ms: performance.now() - started, decode_tps: decode, completion_tokens: n, token_latency_reason: reason, backend_telemetry: telemetry };
}