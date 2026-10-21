<script lang="ts">
  import { onMount } from 'svelte';
  export let metrics: { ttft_ms: number | null; total_ms: number; decode_tps: number | null } | null;
  let el: HTMLDivElement;
  let chart: any;
  onMount(() => {
    let disposed = false;
    (async () => {
      const echarts = await import('echarts');
      if (disposed) return;
      chart = echarts.init(el, 'dark');
      const values = [{ name: 'TTFT', value: metrics?.ttft_ms ?? 0 }, { name: 'Visible total', value: metrics?.total_ms ?? 0 }];
      chart.setOption({ backgroundColor: 'transparent', grid: { left: 8, right: 8, top: 20, bottom: 8, containLabel: true }, xAxis: { type: 'value', axisLabel: { formatter: '{value} ms' } }, yAxis: { type: 'category', data: values.map((v) => v.name) }, series: [{ type: 'bar', data: values.map((v) => v.value), itemStyle: { color: '#7c9cff' } }] });
    })();
    const ro = new ResizeObserver(() => chart?.resize()); ro.observe(el);
    return () => { disposed = true; ro.disconnect(); chart?.dispose(); };
  });
</script>
<div bind:this={el} class="mt-3 h-40 w-full" aria-label="TTFT and total duration chart"></div>
