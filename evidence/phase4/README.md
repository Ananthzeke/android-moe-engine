# Phase 4 evidence

## Real 35B storage-backed run

Artifact: `ggml-org/Qwen3.6-35B-A3B-GGUF`, `Qwen3.6-35B-A3B-Q4_K_M.gguf`, 20,419,565,568
bytes, SHA-256 `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7`.
qwen35moe, 40 layers, 256 experts, top-8 routing, 20,408,576,512 tensor bytes.

- [Aggregate](qwen36-stream-aggregate.json) and [Markdown](qwen36-stream-aggregate.md).
- [Routing reuse](routing-reuse.json) and [cache experiment](cache-experiment.md):
  a 1,024-slot cache reduced reads 23.5% but was slower, so 64-slot demand reads
  remain the default.
- Raw runs [01](qwen36-stream-01.json), [02](qwen36-stream-02.json),
  [03](qwen36-stream-03.json), [04](qwen36-stream-04.json), [05](qwen36-stream-05.json), with stderr and probes.
- Five baseline runs, contexts 512/1024/2048, slots 64/48/64, batch 64, greedy.
- Identical token IDs: `11751, 11, 264, 3177, 34756, 364, 1141, 8807`.
- TTFT 8.8-17.8 s, decode 0.37-0.69 tok/s, peak VmHWM 2,159,888-2,190,576 KiB.
- Arena 28,311,552 / 37,748,736 bytes; 9,999 reads and 5,897,650,176 application bytes;
  no cache hits; maximum union 37.
- CPU thermal zones about 36.1 C to 50.6 C around a measured run.
- [Cancellation record](qwen36-cancel-01.json): 7 partial tokens, no error, exit 0;
  [fresh request after cancellation](qwen36-after-cancel-01.json) returned the
  baseline prefix `11751, 11` and the model SHA-256 was unchanged.
- Application read time 16.6-21.4 s of a 22-28 s generation; storage reads are the measured limiter.

No full-resident reference ran. The 20.4 GB artifact does not fit phone memory or the
available host. Parity evidence is the synthetic runtime gate plus identical tokens
across runs and arena sizes. The process peak is not a hard cap.

## Metadata preflight

- [Qwen3.6-35B-A3B index](qwen36-35b-a3b-q4km-index.json) and
  [manifest](qwen36-35b-a3b-q4km.manifest).
- [Qwen3.5-2B dense index](qwen35-2b-q4km-index.json).

The preflight reads metadata and tensor information only. It refuses full-resident
weights and computes a 2.30 GB streaming estimate; the measured peak was about
2.16 GiB, but that agreement is an observation, not a guarantee.