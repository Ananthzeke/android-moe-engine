import { afterEach, describe, expect, it, vi } from 'vitest';
import { GET, POST } from '../src/routes/api/backend/[...path]/+server';

// Only the request/params boundary is consumed by this route; other Kit services are unused.
function event(path: string, method = 'GET') {
  return { params: { path }, request: new Request('http://localhost/api/backend/' + path, { method }) } as Parameters<typeof GET>[0];
}
afterEach(() => vi.unstubAllGlobals());
describe('same-origin backend proxy', () => {
  it('returns actionable 502 when the phone is unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('connection refused')));
    const response = await GET(event('health'));
    expect(response.status).toBe(502);
    expect((await response.json()).error.code).toBe('backend_unreachable');
  });
  it('rejects unrelated paths and mismatched methods before contacting the phone', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    expect((await GET(event('private'))).status).toBe(403);
    expect((await GET(event('v1/chat/completions'))).status).toBe(405);
    expect((await POST(event('health', 'POST'))).status).toBe(405);
    expect(fetch).not.toHaveBeenCalled();
  });
  it('preserves the upstream streaming body, overload status, and retry hint', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('busy', { status: 429, headers: { 'retry-after': '3' } })));
    const response = await POST(event('v1/chat/completions', 'POST'));
    expect(response.status).toBe(429);
    expect(response.headers.get('retry-after')).toBe('3');
    expect(await response.text()).toBe('busy');
  });
});
