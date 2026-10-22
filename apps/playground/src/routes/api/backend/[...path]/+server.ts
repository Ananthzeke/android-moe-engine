import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const ALLOWED = (process.env.AME_ALLOWED_BACKENDS || 'http://127.0.0.1:8080').split(',').map((x) => x.trim().replace(/\/$/, ''));
const BACKEND = (process.env.AME_BACKEND_URL || ALLOWED[0]).replace(/\/$/, '');
const PATHS: Record<string, string> = { health: 'GET', 'v1/models': 'GET', 'v1/chat/completions': 'POST' };

const proxy: RequestHandler = async ({ params, request }) => {
  const path = params.path || '';
  if (!ALLOWED.includes(BACKEND) || !Object.hasOwn(PATHS, path)) {
    return json({ error: { message: 'Backend destination or path not allowed.' } }, { status: 403 });
  }
  if (PATHS[path] !== request.method) {
    return json({ error: { message: 'Method not allowed for this backend path.' } }, { status: 405, headers: { allow: PATHS[path] } });
  }
  const body = request.method === 'POST' ? await request.text() : undefined;
  let upstream: Response;
  try {
    upstream = await fetch(new URL(path, BACKEND + '/'), {
      method: request.method, signal: request.signal,
      headers: { 'content-type': 'application/json', accept: request.headers.get('accept') || 'application/json' }, body
    });
  } catch {
    return json({ error: { code: 'backend_unreachable', message: 'Phone API unreachable. Check adb devices, the port forward, and that the phone server is running; then select Reconnect.' } }, { status: 502 });
  }
  const headers = new Headers({ 'content-type': upstream.headers.get('content-type') || 'application/json', 'cache-control': 'no-store', 'x-accel-buffering': 'no' });
  const retryAfter = upstream.headers.get('retry-after');
  if (retryAfter) headers.set('retry-after', retryAfter);
  return new Response(upstream.body, { status: upstream.status, headers });
};
export const GET = proxy;
export const POST = proxy;
