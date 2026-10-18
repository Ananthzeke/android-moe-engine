# Coding rules

- Keep C++20 project code in its owning module. Use two spaces for C++, CMake, and JSON; four for Python. Apply `.clang-format` through the verifier. Do not add ISA assumptions or `-march=native`.
- Name domain values: offsets and byte counts are distinct types. Validate external counts, arithmetic, file bounds, layouts, and identity before allocation or execution. Prefer subtraction for containment after validating the offset.
- `CheckedFileExtent` proves containment against a supplied size snapshot only. Empty extents, including EOF, are valid. Required tensor payloads must separately enforce nonempty data, alignment, quantization blocks, strides, hashes, and immutable identity.
- Use RAII and explicit owners. Borrowed views cannot outlive storage or asynchronous consumers. State which thread owns mutable state; cancellation must drain consumers before teardown. One active generation is the MVP contract.
- Reserve bytes before materialization and account for alignment, staging, metadata, copies, packing, and overlapping state. Report accounting coverage separately from RSS/PSS and OS cache effects.
- Keep backend mathematics and tokenizer semantics upstream. Preserve a small identifiable patch against one pinned backend after the feasibility gate. Do not invent engine APIs or generic frameworks ahead of evidence.
- Keep validation internals private; no installed SDK exists. Prefer explicit small error values at boundaries. Do not use assertions for required runtime validation or the only test checks.
- Test observable success, boundary rejection, failure, and lifetime behavior. Debug, Release, and host sanitizer tests must pass before handoff. Do not add tests that only repeat implementation details.
- Return nonzero on tool/test failure. Unsupported observations use null plus a reason; retain failed runs. Require provenance and token evidence before promoting model support or performance claims.

[Contribution commands](../CONTRIBUTING.md) define the authoritative offline verifier. Compiler warnings apply to project targets only, including conversion, sign conversion, shadowing, and warnings as errors on GCC/Clang. Do not leak these flags into future upstream builds.
