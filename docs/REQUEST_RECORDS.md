# Request and startup records

The native server can record where time and bytes go for every request and for startup.
Pass `--request-log PATH` (phone launcher: `AME_REQUEST_LOG=PATH`) to append JSON lines.
The first line is the startup record; each request adds one line. `GET /health` carries the
same startup record and a bounded copy of the last request. A request body may add
`"ame_debug": true` to receive its record in the JSON response under `ame`.
`tools/run_record.py --request-log PATH` summarizes a log into `result.json`, and
`tools/request_series.py` sends a fixed series and keeps every record.

## What each field means

- **Phases.** A request has `context_create`, `prefill`, `decode`, and `context_teardown`
  phases. Each has wall `seconds` from a monotonic clock. Tokenization and chat rendering
  happen before admission and are not in these phases. `queue_wait_seconds` is the time
  spent waiting for the generation slot; `total_seconds` runs from the handler start.
- **Process and system counters** (`proc_delta`), as differences between samples taken at
  the phase start and end:
  - `rchar`: bytes the process asked read calls for. It includes page-cache hits.
  - `read_bytes`: bytes the block layer fetched on behalf of the process. It includes
    mmap page faults and readahead. The two are separate quantities. Their ratio is not a
    cache-hit rate, and the server never computes it.
  - `majflt`: major page faults of this process.
  - `pswpin`: pages swapped in, system-wide (`/proc/vmstat`). It is not limited to this
    process.
- **Expert counters** (`expert_delta`) are differences of the lifetime counters, so a
  failed or cancelled request does not leak into the next one: `hits`, `misses` (a slot
  that was not resident when requested, counted when the slot is reserved, so a read that
  then fails still counts), `expert_reads` (expert slices read, not syscalls),
  `expert_bytes` (application bytes for those slices), `pread_calls` and `pread_bytes` (the
  actual syscalls and the bytes they returned), and `read_seconds` (wall time inside the
  file source; not physical flash time).
- **Big-core frequency.** The big cores are the CPUs whose `cpuinfo_max_freq` equals the
  highest value; they are found at startup, never hard-coded. `big_core_khz_start` and
  `big_core_khz_end` are `scaling_cur_freq` at the phase boundaries. A system whose cores
  all share one maximum is flagged `homogeneous`. These are two instants, not a clock
  history, and a frequency can change within a phase.
- **Tokens.** `prompt_token_ids` and `token_ids` are the exact IDs: every sampled token,
  including a final end-of-generation token (`ended_with_eog`). `decode_step_seconds` holds
  the `llama_decode` time after each emitted token.
- **Unavailable values.** A counter that cannot be read is `null`, and `unavailable` lists
  its field and reason (`cannot_open`, `field_missing`, `field_malformed`,
  `counter_decreased`). A zero always means a measured zero.

## Startup stages

`first_ready_seconds` runs from the start of `main` to readiness. `stages` times, one after
another: `artifact_open` (opens the file and hashes all of it; `artifact_hash_seconds` is
the hashing alone), `manifest_install` (`manifest_validation_seconds` is the manifest and
GGUF-table check alone), `model_load`, `contract_bind`, and `template_init`. Each stage
carries its own `proc_delta`, so the bytes read during hashing and during model load are
attributed separately.

## Status values

`completed`, `failed` (with `error`), `cancelled`, or `deadline_exceeded`. The deadline is
checked between prefill chunks and decode steps; a blocked read can exceed it.
