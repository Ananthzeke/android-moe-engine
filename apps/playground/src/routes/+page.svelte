<script lang="ts">
  import { onMount } from 'svelte';
  import { marked } from 'marked';
  import DOMPurify from 'dompurify';
  import { streamChat, getHealth, getModels, type Health, type ChatMessage, type GenerationMetrics } from '$lib/api';
  import { saveConversation, listConversations, clearConversations, type Conversation } from '$lib/db';
  import Chart from '$lib/Chart.svelte';

  let messages: ChatMessage[] = [];
  let system = 'You are a concise assistant running locally on an Android phone.';
  let model = '';
  let models: string[] = [];
  let temperature = 0;
  let maxTokens = 128;
  let seed = 1234;
  let health: Health | null = null;
  let error = '';
  let status = 'idle';
  let storageWarning = '';
  let snapshotAt: string | null = null;
  $: connected = health?.status === 'ready' && models.includes(model);
  let controller: AbortController | null = null;
  let streaming = false;
  let lastMetrics: GenerationMetrics | null = null;
  let history: Conversation[] = [];
  let showStats = false;
  let showSettings = true;

  function fmt(n: number | null | undefined, suffix = '') { return n == null || !Number.isFinite(n) ? 'unavailable' : n.toFixed(2) + suffix; }
  function counter(value: unknown, divisor = 1, suffix = '') { return typeof value === 'number' ? (divisor === 1 ? value.toLocaleString() : fmt(value / divisor, suffix)) : 'unavailable'; }
  function render(text: string) { return DOMPurify.sanitize(marked.parse(text || '') as string); }
  async function refresh() {
    status = 'connecting'; error = '';
    try {
      const nextHealth = await getHealth();
      const nextModels = await getModels();
      health = nextHealth; models = nextModels;
      if (!models.includes(model)) model = models[0] ?? '';
      snapshotAt = new Date().toLocaleTimeString();
    }
    catch (e) { health = null; models = []; model = ''; snapshotAt = null; status = 'failed'; error = e instanceof Error ? e.message : String(e); }
    if (status === 'connecting') status = 'idle';
  }
  onMount(() => {
    void refresh();
    void listConversations().then((saved) => { history = saved; }).catch(() => { storageWarning = 'Browser history is unavailable. Chat still works; export any responses you want to keep.'; });
  });
  async function send(text: string, appendUser = true) {
    if (streaming || !connected || !text.trim()) return;
    error = ''; lastMetrics = null;
    messages = appendUser ? [...messages, { role: 'user', content: text }, { role: 'assistant', content: '', status: 'waiting' }] : [...messages, { role: 'assistant', content: '', status: 'waiting' }];
    const assistant = messages[messages.length - 1];
    streaming = true; status = 'waiting'; controller = new AbortController();
    const context = [{ role: 'system' as const, content: system }, ...messages.slice(0, -1).filter((m) => m.content.trim())];
    try {
      const result = await streamChat(context, { model, temperature, max_tokens: maxTokens, seed, stream: true, stream_options: { include_usage: true } }, controller.signal, (delta, metrics, phase) => {
        if (phase === 'waiting') { status = 'waiting'; assistant.status = 'waiting'; }
        if (metrics) lastMetrics = metrics;
        if (delta) { assistant.content += delta; assistant.status = 'generating'; status = 'generating'; messages = messages; }
      });
      assistant.content = result.content; assistant.metrics = result.metrics; assistant.usage = result.usage; assistant.status = 'completed';
      lastMetrics = result.metrics; status = 'completed';
    } catch (e) {
      if (controller?.signal.aborted) { assistant.status = 'cancelled'; status = 'cancelled'; }
      else { assistant.status = 'failed'; status = 'failed'; error = e instanceof Error ? e.message : String(e); }
    } finally {
      streaming = false; controller = null; messages = messages;
      if (messages.length) {
        const c: Conversation = { id: crypto.randomUUID(), title: messages.find((m) => m.role === 'user')?.content.slice(0, 80) || 'Untitled', model, system, messages: messages.map((m) => ({ ...m })), metrics: lastMetrics, createdAt: Date.now() };
        history = [c, ...history].slice(0, 50);
        try { await saveConversation(c); } catch { storageWarning = 'Could not save browser history. Export this response to keep it.'; }
      }
    }
  }
  function stop() { if (controller) { status = 'cancelling'; controller.abort(); } }
  function clear() { if (!streaming) { messages = []; lastMetrics = null; error = ''; status = 'idle'; } }
  function retry() { if (streaming) return; const idx = messages.map((m) => m.role).lastIndexOf('user'); if (idx < 0) return; const text = messages[idx].content; messages = messages.slice(0, idx + 1); send(text, false); }
  function copy() { const a = [...messages].reverse().find((m) => m.role === 'assistant'); if (a) navigator.clipboard.writeText(a.content); }
  function dl(content: string, name: string, type: string) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([content], { type })); a.download = name; a.click(); URL.revokeObjectURL(a.href); }
  function exportJson() { dl(JSON.stringify({ model, system, temperature, maxTokens, seed, messages, metrics: lastMetrics, createdAt: new Date().toISOString() }, null, 2), 'android-moe-chat.json', 'application/json'); }
  function exportCsv() { const rows = [['role','content','completion_tokens','ttft_ms','total_ms','decode_tps'], ...messages.map((m) => [m.role, m.content, m.usage?.completion_tokens ?? '', m.metrics?.ttft_ms ?? '', m.metrics?.total_ms ?? '', m.metrics?.decode_tps ?? ''])]; dl(rows.map((r) => r.map((x) => '"' + String(x).replaceAll('"','""') + '"').join(',')).join('\n'), 'android-moe-chat.csv', 'text/csv'); }
  async function wipe() { try { await clearConversations(); history = []; storageWarning = ''; } catch { storageWarning = 'Could not clear browser history.'; } }
</script>
<svelte:head><title>Playground · Android MoE Engine</title></svelte:head>
<main class="mx-auto flex min-h-screen max-w-[1500px] flex-col gap-4 p-4 lg:p-6">
  <header class="flex flex-wrap items-center justify-between gap-3"><div><h1 class="text-2xl font-semibold tracking-tight">Android MoE <span class="text-accent">Playground</span></h1><p class="text-sm text-slate-400">Local UI · inference stays on the phone · experimental v0.1.0-alpha.1</p></div><div class="flex items-center gap-3"><span role="status" class="rounded-full border px-3 py-1 text-sm {connected ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-200' : 'border-amber-400/30 bg-amber-400/10 text-amber-200'}">{connected ? 'Connected' : 'Disconnected'} · {status}</span><button class="btn" onclick={refresh} disabled={streaming}>Reconnect</button></div></header>
  {#if error}<div role="alert" class="rounded-xl border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-200">{error}</div>{/if}
  <div class="grid gap-4 xl:grid-cols-[260px_minmax(0,1fr)_280px]">
    <aside class="card space-y-3 p-4 order-2 xl:order-1"><div class="flex items-center justify-between"><h2 class="font-semibold">Connection</h2><button class="text-xs text-accent" onclick={() => showSettings = !showSettings}>settings</button></div><p class="text-xs text-slate-400">Proxy <code>/api/backend</code> → ADB-forwarded phone.</p>
      {#if health}<dl class="space-y-1 text-sm"><div class="flex justify-between"><dt>Status</dt><dd class={health.status === 'ready' ? 'text-emerald-300' : 'text-amber-300'}>{health.status}</dd></div><div class="flex justify-between gap-3"><dt>Model</dt><dd class="min-w-0 break-words text-right">{health.model}</dd></div>{#if health.queue}<div class="flex justify-between"><dt>Outstanding</dt><dd>{health.queue.outstanding}/4</dd></div><div class="flex justify-between"><dt>Active</dt><dd>{health.queue.active ? 'yes' : 'no'}</dd></div>{/if}</dl>{:else}<p class="text-sm text-slate-400">No health snapshot.</p>{/if}
      {#if showSettings}<label class="label" for="model">Loaded model</label><select id="model" class="input" bind:value={model} disabled={streaming || !models.length}>{#each models as m}<option value={m}>{m}</option>{/each}</select><p class="text-xs text-slate-500">Only the server-reported loaded model. This UI never downloads or switches weights.</p>
      <label class="label" for="system">System prompt</label><textarea id="system" class="input h-24" bind:value={system} disabled={streaming}></textarea>
      <div class="grid grid-cols-2 gap-2"><div><label class="label" for="temp">Temperature</label><input id="temp" class="input" type="number" min="0" max="2" step="0.1" bind:value={temperature} disabled={streaming} /></div><div><label class="label" for="max">Max tokens</label><input id="max" class="input" type="number" min="1" max="4096" bind:value={maxTokens} disabled={streaming} /></div></div>
      <label class="label" for="seed">Seed</label><input id="seed" class="input" type="number" bind:value={seed} disabled={streaming} />{/if}
      <p class="border-t border-white/10 pt-3 text-xs text-slate-500">Context, threads, and runtime mode are not reported by this API. Check the phone server launch configuration.</p>
      {#if snapshotAt}<p class="text-xs text-slate-500">Health snapshot at {snapshotAt}. Reconnect refreshes it.</p>{/if}
    </aside>
    <section class="card order-1 flex min-h-[65vh] min-w-0 flex-col p-4 xl:order-2 xl:h-[calc(100vh-10rem)]"><div class="mb-3 flex flex-wrap items-center justify-between gap-2"><h2 class="font-semibold">Chat</h2><div class="flex flex-wrap gap-2"><button class="btn" onclick={copy} disabled={!messages.some((m) => m.role === 'assistant' && m.content)}>Copy</button><button class="btn" onclick={retry} disabled={streaming || !messages.some((m) => m.role === 'user')}>Retry</button><button class="btn" onclick={clear} disabled={streaming}>Clear</button><button class="btn" onclick={exportJson} disabled={!messages.length}>JSON</button><button class="btn" onclick={exportCsv} disabled={!messages.length}>CSV</button></div></div>
      <div class="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">{#if !messages.length}<div class="grid h-full place-items-center text-center text-slate-400"><div><p class="text-lg">Start a local conversation.</p><p class="text-sm">The phone runs inference. This laptop proxies and stores history.</p></div></div>{/if}
        {#each messages as message}<article class="max-w-[95%] rounded-2xl border border-white/10 p-3 {message.role === 'user' ? 'ml-auto bg-accent/10' : 'bg-black/20'}" class:opacity-70={message.status === 'cancelled'}><div class="mb-1 flex items-center gap-2 text-xs uppercase tracking-wider text-slate-400"><span>{message.role}</span>{#if message.status}<span>· {message.status}</span>{/if}</div><div class="markdown max-w-none break-words text-sm" class:whitespace-pre-wrap={message.role !== 'assistant'}>{#if message.role === 'assistant'}{@html render(message.content || (message.status === 'waiting' ? 'Waiting for the phone…' : ''))}{:else}{message.content}{/if}</div>{#if message.role === 'assistant' && message.metrics}<div class="mt-3 grid grid-cols-2 gap-2 border-t border-white/10 pt-2 text-xs text-slate-400 md:grid-cols-4"><span>TTFT <b class="text-slate-200">{fmt(message.metrics.ttft_ms, ' ms')}</b></span><span>Total <b class="text-slate-200">{fmt(message.metrics.total_ms, ' ms')}</b></span><span>Delivery <b class="text-slate-200">{fmt(message.metrics.decode_tps, ' tok/s')}</b></span><span>Tokens <b class="text-slate-200">{message.usage?.completion_tokens ?? 'unavailable'}</b></span></div>{/if}</article>{/each}</div>
      <form class="mt-4 flex gap-2" onsubmit={(e) => { e.preventDefault(); const f = e.currentTarget; const input = f.elements.namedItem('prompt') as HTMLTextAreaElement; const v = input.value; input.value = ''; send(v); }}><textarea aria-label="Message" name="prompt" disabled={!connected || streaming} class="input h-20 resize-none" placeholder="Ask the phone… (Enter sends, Shift+Enter newline)" onkeydown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); } }}></textarea>{#if streaming}<button type="button" class="btn border-red-400/40 text-red-200" onclick={stop}>Stop</button>{:else}<button class="btn-primary" type="submit" disabled={!connected}>Send</button>{/if}</form>
    </section>
    <aside class="order-3 space-y-4"><section class="card p-4"><div class="flex items-center justify-between"><h2 class="font-semibold">Response metrics</h2><button class="text-xs text-accent" onclick={() => showStats = !showStats}>{showStats ? 'hide' : 'chart'}</button></div>{#if lastMetrics}<dl class="mt-3 space-y-2 text-sm"><div class="flex justify-between"><dt>Browser TTFT</dt><dd>{fmt(lastMetrics.ttft_ms, ' ms')}</dd></div><div class="flex justify-between"><dt>Request total</dt><dd>{fmt(lastMetrics.total_ms, ' ms')}</dd></div><div class="flex justify-between"><dt>Text delivery rate</dt><dd>{fmt(lastMetrics.decode_tps, ' tok/s')}</dd></div><div class="flex justify-between"><dt>Generated</dt><dd>{lastMetrics.completion_tokens ?? 'unavailable'}</dd></div><div class="flex justify-between"><dt>Token latency</dt><dd>unavailable</dd></div></dl>{:else}<p class="mt-2 text-sm text-slate-400">No completed response yet.</p>{/if}{#if lastMetrics}<p class="mt-2 text-xs text-slate-500">Browser timings include proxy, transport, and queue time. Delivery = (N−1)/(last−first visible text); buffering can skew it. Engine decode speed and per-token latency are unavailable.</p>{/if}{#if showStats}<Chart metrics={lastMetrics} />{/if}</section>
      <section class="card p-4"><h2 class="font-semibold">Expert cache snapshot</h2>{#if health?.streaming}<dl class="mt-3 grid grid-cols-2 gap-2 text-sm"><div><dt class="text-slate-400">Arena</dt><dd>{counter(health.streaming.arena_bytes, 1048576, ' MiB')}</dd></div><div><dt class="text-slate-400">Hits</dt><dd>{counter(health.streaming.hits)}</dd></div><div><dt class="text-slate-400">Misses</dt><dd>{counter(health.streaming.misses)}</dd></div><div><dt class="text-slate-400">Evictions</dt><dd>{counter(health.streaming.evictions)}</dd></div><div><dt class="text-slate-400">App reads</dt><dd>{counter(health.streaming.read_bytes, 1e9, ' GB')}</dd></div><div><dt class="text-slate-400">Read calls</dt><dd>{counter(health.streaming.reads)}</dd></div></dl><p class="mt-2 text-xs text-slate-500">Cumulative process counters. Read bytes are application reads, not physical flash I/O.</p>{:else}<p class="mt-2 text-sm text-slate-400">Streaming counters unavailable. This does not establish runtime mode.</p>{/if}</section>
      <section class="card p-4"><div class="flex items-center justify-between"><h2 class="font-semibold">History</h2><button class="text-xs text-red-300" onclick={wipe}>clear</button></div><div class="mt-2 max-h-64 space-y-2 overflow-y-auto">{#each history.slice(0, 8) as h}<div class="rounded-lg bg-white/5 p-2 text-xs"><p class="truncate text-slate-200">{h.title}</p><p class="text-slate-500">{h.model} · {new Date(h.createdAt).toLocaleString()}</p></div>{/each}{#if !history.length}<p class="text-sm text-slate-400">No saved conversations.</p>{/if}</div><p class="mt-2 text-xs text-slate-500">Stored in this browser. Prompts are sent only to the configured backend.</p></section>
    </aside>
  </div>
  {#if storageWarning}<p role="status" class="text-sm text-amber-200">{storageWarning}</p>{/if}
  <footer class="flex flex-wrap justify-between gap-2 text-xs text-slate-500"><span>Experimental · local-only · no external providers</span><span>No automatic model loading</span></footer>
</main>