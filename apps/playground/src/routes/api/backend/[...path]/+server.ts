import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
const ALLOWED = (process.env.AME_ALLOWED_BACKENDS || 'http://127.0.0.1:8080').split(',').map((x: string) => x.trim().replace(/\/$/, ''));
const BACKEND = (process.env.AME_BACKEND_URL || ALLOWED[0]).replace(/\/$/, '');
const PATHS = new Set(['health','v1/models','v1/chat/completions']);
function target(path: string): URL | null {
  if (!ALLOWED.includes(BACKEND) || !PATHS.has(path)) return null;
  return new URL(path, BACKEND + '/');
}
export const GET: RequestHandler = async ({ params, request }) => {
  const url = target(params.path || '');
  if (!url) return json({error:{message:'backend destination or path not allowed'}}, {status:403});
  const r = await fetch(url, {signal:request.signal, headers:{accept:request.headers.get('accept')||'application/json'}});
  return new Response(r.body,{status:r.status,headers:{'content-type':r.headers.get('content-type')||'application/json','cache-control':'no-store'}});
};
export const POST: RequestHandler = async ({ params, request }) => {
  const url = target(params.path || '');
  if (!url) return json({error:{message:'backend destination or path not allowed'}}, {status:403});
  const r = await fetch(url,{method:'POST',signal:request.signal,headers:{'content-type':'application/json',accept:request.headers.get('accept')||'text/event-stream'},body:await request.text()});
  return new Response(r.body,{status:r.status,headers:{'content-type':r.headers.get('content-type')||'application/json','cache-control':'no-cache','x-accel-buffering':'no'}});
};