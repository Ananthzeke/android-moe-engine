#!/bin/sh
# Generate a tiny MoE fixture (default qwen35moe; pass qwen2moe as the fourth argument for
# separate gate, up and down expert tensors) with the upstream architecture test.
# The fixture is deterministic for a fixed pinned tree, seed, and stdev.
# AME_FIXTURE_EXPERTS=N (optional) builds the generator from a scratch copy of the source with N
# experts and top-2 routing instead of the upstream 2, so a small pool must evict and the router
# has to choose. The pinned tree itself is not edited.
set -eu
: "${AME_LLAMA_SOURCE:?set AME_LLAMA_SOURCE to the pinned llama.cpp tree}"
: "${AME_LLAMA_BUILD:?set AME_LLAMA_BUILD to a build directory}"
OUT=${1:?usage: generate_qwen35_fixture.sh OUT_DIR}
SEED=${2:-1234}
STDEV=${3:-0.1}
ARCH=${4:-qwen35moe}
mkdir -p "$OUT"
if [ -n "${AME_FIXTURE_EXPERTS:-}" ]; then
  SPARSE_SRC="$AME_LLAMA_BUILD-src"
  rm -rf "$SPARSE_SRC"
  mkdir -p "$SPARSE_SRC"
  tar -C "$AME_LLAMA_SOURCE" --exclude=.git -cf - . | tar -C "$SPARSE_SRC" -xf -
  sed -i "s/uint32_t(host_experts_test(arch) ? 64 : 2));/uint32_t(host_experts_test(arch) ? 64 : $AME_FIXTURE_EXPERTS));/" \
    "$SPARSE_SRC/tests/test-llama-archs.cpp"
  grep -q "host_experts_test(arch) ? 64 : $AME_FIXTURE_EXPERTS" "$SPARSE_SRC/tests/test-llama-archs.cpp" ||
    { echo "expert count line not found in test-llama-archs.cpp" >&2; exit 2; }
  AME_LLAMA_SOURCE=$SPARSE_SRC
fi
cmake -S "$AME_LLAMA_SOURCE" -B "$AME_LLAMA_BUILD" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build "$AME_LLAMA_BUILD" --target test-llama-archs -j 6
"$AME_LLAMA_BUILD/bin/test-llama-archs" -a "$ARCH" -o "$OUT" -s "$SEED" -d "$STDEV"
