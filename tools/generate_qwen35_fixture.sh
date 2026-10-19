#!/bin/sh
# Generate the tiny qwen35moe fixture with the upstream architecture test.
# The fixture is deterministic for a fixed pinned tree, seed, and stdev.
set -eu
: "${AME_LLAMA_SOURCE:?set AME_LLAMA_SOURCE to the pinned llama.cpp tree}"
: "${AME_LLAMA_BUILD:?set AME_LLAMA_BUILD to a build directory}"
OUT=${1:?usage: generate_qwen35_fixture.sh OUT_DIR}
SEED=${2:-1234}
STDEV=${3:-0.1}
mkdir -p "$OUT"
cmake -S "$AME_LLAMA_SOURCE" -B "$AME_LLAMA_BUILD" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build "$AME_LLAMA_BUILD" --target test-llama-archs -j 6
"$AME_LLAMA_BUILD/bin/test-llama-archs" -a qwen35moe -o "$OUT" -s "$SEED" -d "$STDEV"
