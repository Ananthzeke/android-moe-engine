<script lang="ts">
  import { onMount } from 'svelte';
  import type { ECharts } from 'echarts';
  export let metrics: { ttft_ms: number | null; total_ms: number; decode_tps: number | null } | null;
  let el: HTMLDivElement;
  let chart: ECharts | undefined;
  function update() {
    const values = [{ name: 'Browser TTFT', value: metrics?.ttft_ms ?? null }, { name: 'Request total', value: metrics?.total_ms ?? null }];
    chart?.setOption({ backgroundColor: 'transparent', grid: { left: 8, right: 8, top: 20, bottom: 8, containLabel: true }, xAxis: { type: 'value', splitNumber: 2, axisLabel: { hideOverlap: true, formatter: (value: number) => (value / 1000).toFixed(1) + ' s' } }, yAxis: { type: 'category', data: values.map((v) => v.name) }, series: [{ type: 'bar', data: values.map((v) => v.value), itemStyle: { color: '#7c9cff' } }] });
  }
  $: if (chart) { metrics; update(); }
  onMount(() => {
    let disposed = false;
    void import('echarts').then((echarts) => {
      if (disposed) return;
      chart = echarts.init(el, 'dark');
      update();
    });
    const ro = new ResizeObserver(() => chart?.resize()); ro.observe(el);
    return () => { disposed = true; ro.disconnect(); chart?.dispose(); };
  });
</script>
{#if !metrics}<p class="mt-3 text-xs text-slate-500">Complete a response to plot browser timings.</p>{/if}
<div bind:this={el} class="mt-3 h-40 w-full" aria-label="Browser TTFT and request duration chart"></div>
