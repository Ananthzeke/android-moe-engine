import { describe, expect, it, vi, afterEach } from 'vitest';
import { streamChat } from '../src/lib/api';
function chunkStream(chunks: string[]) {
  const enc = new TextEncoder();
  return new ReadableStream({ start(c) { for (const chunk of chunks) c.enqueue(enc.encode(chunk)); c.close(); } });
}
function sse(event: unknown) { return 'data: ' + JSON.stringify(event) + '\n\n'; }
function mock(chunks: string[], status = 200) {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(chunkStream(chunks), { status, headers: { 'content-type': 'text/event-stream' } })));
}
const base = { model: 'm', temperature: 0, max_tokens: 8, seed: 1, stream: true };
afterEach(() => vi.unstubAllGlobals());
describe('SSE chat client', () => {
  it('handles split events, multiple events per read, role-only, UTF-8 and usage', async () => {
    mock([
      sse({ choices: [{ delta: { role: 'assistant' } }] }).slice(0, 8),
      sse({ choices: [{ delta: { role: 'assistant' } }] }).slice(8) + sse({ choices: [{ delta: { content: 'hé' } }] }),
      sse({ choices: [{ delta: { content: 'llo' } }] }) + sse({ choices: [{ finish_reason: 'stop', delta: {} }], usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 } }) + 'data: [DONE]\n\n'
    ]);
    const seen: string[] = [];
    const r = await streamChat([{ role: 'user', content: 'hi' }], base, new AbortController().signal, (d) => { if (d) seen.push(d); });
    expect(r.content).toBe('héllo');
    expect(seen.join('')).toBe('héllo');
    expect(r.usage?.completion_tokens).toBe(2);
    expect(r.finishReason).toBe('stop');
    expect(r.metrics.ttft_ms).not.toBeNull();
  });
  it('returns null decode for fewer than two tokens with a reason', async () => {
    mock([sse({ choices: [{ delta: { content: 'x' } }] }), sse({ choices: [{ finish_reason: 'length', delta: {} }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } })]);
    const r = await streamChat([{ role: 'user', content: 'hi' }], base, new AbortController().signal, () => {});
    expect(r.metrics.decode_tps).toBeNull();
    expect(r.metrics.token_latency_reason).toContain('fewer');
  });
  it('returns null decode when usage is missing', async () => {
    mock([sse({ choices: [{ delta: { content: 'a' } }] }), sse({ choices: [{ delta: { content: 'b' } }] }), sse({ choices: [{ finish_reason: 'stop', delta: {} }] })]);
    const r = await streamChat([{ role: 'user', content: 'hi' }], base, new AbortController().signal, () => {});
    expect(r.metrics.decode_tps).toBeNull();
    expect(r.metrics.token_latency_reason).toContain('usage');
  });
  it('rejects malformed SSE JSON', async () => {
    mock(['data: {not json}\n\n']);
    await expect(streamChat([{ role: 'user', content: 'hi' }], base, new AbortController().signal, () => {})).rejects.toThrow('malformed');
  });
  it('surfaces 429 without inventing a response', async () => {
    mock([JSON.stringify({ error: { message: 'four requests are already outstanding', code: 'server_busy' } })], 429);
    await expect(streamChat([{ role: 'user', content: 'hi' }], base, new AbortController().signal, () => {})).rejects.toThrow('four requests');
  });
});
