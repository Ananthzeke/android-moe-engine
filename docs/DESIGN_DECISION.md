# Design decision

Date: 18 October 2026. Scope: the initial design and DeepSeek handoff, not implementation approval of an untested engine.

## Chosen direction

Build a CPU engine around one pinned backend fork with a private routed-expert integration. Keep tokenization, graph mathematics, and kernels in that backend. Own expert storage, accounting, residency, leases, cancellation, and evidence here. Evaluate MNN first because inspected code exposes an expert-module boundary. Select the production backend only after the phase 1 source experiment.

The small public surface lets a caller inspect, open, and generate. The caller does not coordinate file loading, expert cache slots, or backend tensors.

## Alternatives considered

| Shape | Decision and cost |
| --- | --- |
| Standalone runtime and new universal model format | Rejected for MVP. Kernel, tokenizer, conversion, hybrid-state, and numerical work would obscure the storage experiment. |
| Unmodified runtime with mmap | Retained as baseline. It delegates residency to the OS and does not establish our bounded expert-memory ownership. |
| Multiple production backend adapters | Deferred. Integration experiments may compare two, but the MVP maintains one. |
| Backend fork with explicit expert lifetime | Selected provisionally. It provides the needed control at the cost of a small upstream patch and source-level feasibility risk. |

## Independent designs and synthesis

Three independent candidates used the configured architect models: gpt-6.1-sol, gpt-6-sol, and gpt-6-luna. Each compared whole architecture shapes and recommended a gated backend fork. A separate gpt-6-luna cross-judge evaluated the completed designs. The primary author read every candidate and agreed with its selected base.

Scores are design-review judgments from 1 to 5, not performance measurements:

| Candidate | Integration gate | Memory and I/O | Model correctness | Evidence | Interface and scope |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpt-6.1-sol | 5 | 4 | 4 | 5 | 5 |
| gpt-6-sol | 4 | 4 | 4 | 5 | 3 |
| gpt-6-luna | 4 | 3 | 3 | 4 | 2 |

Use the gpt-6.1-sol candidate as the base for its single owned budget and narrow public interface. Adapt the gpt-6-sol candidate's explicit source-experiment matrix and interruption records. Retain the consensus outcome that a failed streaming gate still leaves a verified small CPU engine. Reject broader installation, Android UI, and runtime-plugin APIs for the MVP.

Review identified four corrections now included in the architecture:

- Seeing routed expert IDs does not prove that retained expert-module weights can be bounded.
- A decode-sized cache cannot be assumed to fit prefill's union of experts. Prove bounded scheduling or refuse.
- A regular-file read has no guaranteed wall-clock cancellation deadline. Bound chunks and measure latency.
- An arena cap is not a process-memory cap. Backend allocation coverage and OS page-cache behavior remain separate.

## Accepted tradeoffs

We accept upstream patch maintenance to reuse numerical kernels. We accept demand-read latency to keep ownership and I/O interpretable. We accept one-session CPU text scope to establish correctness before optional accelerators and an app interface.

## Unresolved facts and resolution owners

The phase 1 implementer resolves expert backing lifetime, exact hook placement, allocation coverage, minimum live extents, and conversion support. The physical-target evaluator resolves storage wait, compute cost, thermal behavior, and achievable latency. These are experimental questions, not preference questions for the repository owner.

## Implementation reconciliation

No engine changes have been implemented. Future accepted deviations must update the affected architecture, API, phase gates, and this section together. Record the source experiment or reviewer decision that supports each deviation. Do not alter a contract merely to relabel a failing test as success.

The next implementation step is phase 0 followed by one real small-model baseline and the source-level MoE fixture.
