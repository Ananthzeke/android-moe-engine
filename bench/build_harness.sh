#!/bin/sh
# Build the baseline harness and the two expert-boundary gates against the pinned
# llama.cpp. The runtime gate requires patches/0001-ggml-moe-materializer.patch to
# be applied to the pinned llama.cpp tree before building it.
# Required environment:
#   AME_TOOLS   directory containing src/llama.cpp, build/llama-host,
#               build/llama-android, and android-ndk-r27c
# Optional:
#   AME_REPO    repository root (defaults to this script's parent)
#   AME_NDK_API Android API level (default 31)
set -eu

: "${AME_TOOLS:?set AME_TOOLS to the pinned tools directory}"
AME_REPO=${AME_REPO:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
NDK="$AME_TOOLS/android-ndk-r27c"
LLAMA="$AME_TOOLS/src/llama.cpp"
HOST_BUILD="$AME_TOOLS/build/llama-host"
ANDROID_BUILD="$AME_TOOLS/build/llama-android"
OUT="$AME_TOOLS/build/harness"
API=${AME_NDK_API:-31}
CXX="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${API}-clang++"
STRIP="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"
CXXFLAGS="-std=c++20 -O2 -DNDEBUG"
ENGINE="src/backend/moe_stream.cpp src/model/file_extent.cpp src/model/expert_layout.cpp src/residency/file_source.cpp src/residency/store.cpp"
ENGINE_INCLUDES="-Isrc/backend -Isrc/model -Isrc/residency -Isrc/common"
LIBS_HOST="$HOST_BUILD/ggml/src/libggml.a $HOST_BUILD/ggml/src/libggml-base.a $HOST_BUILD/ggml/src/libggml-cpu.a"
LIBS_ANDROID="$ANDROID_BUILD/ggml/src/libggml.a $ANDROID_BUILD/ggml/src/libggml-base.a $ANDROID_BUILD/ggml/src/libggml-cpu.a"

mkdir -p "$OUT"
cd "$AME_REPO"

g++ $CXXFLAGS bench/llama_baseline.cpp $ENGINE $ENGINE_INCLUDES \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -Wl,--start-group "$HOST_BUILD/src/libllama.a" $LIBS_HOST -Wl,--end-group \
  -lpthread -ldl -lm -o "$OUT/llama_baseline_host"

g++ $CXXFLAGS bench/moe_kernel_gate.cpp \
  src/model/file_extent.cpp src/model/expert_layout.cpp \
  src/residency/file_source.cpp src/residency/store.cpp \
  -Isrc/model -Isrc/residency -Isrc/common \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -Wl,--start-group $LIBS_HOST -Wl,--end-group \
  -lpthread -ldl -lm -o "$OUT/moe_kernel_gate_host"

g++ $CXXFLAGS bench/moe_stream_gate.cpp $ENGINE $ENGINE_INCLUDES \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -Wl,--start-group $LIBS_HOST -Wl,--end-group \
  -lpthread -ldl -lm -o "$OUT/moe_stream_gate_host"

"$CXX" $CXXFLAGS -static-libstdc++ bench/llama_baseline.cpp $ENGINE $ENGINE_INCLUDES \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -Wl,--start-group "$ANDROID_BUILD/src/libllama.a" $LIBS_ANDROID -Wl,--end-group \
  -llog -lm -o "$OUT/llama_baseline_android"

"$CXX" $CXXFLAGS -static-libstdc++ bench/moe_kernel_gate.cpp \
  src/model/file_extent.cpp src/model/expert_layout.cpp \
  src/residency/file_source.cpp src/residency/store.cpp \
  -Isrc/model -Isrc/residency -Isrc/common \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -Wl,--start-group $LIBS_ANDROID -Wl,--end-group -llog -lm \
  -o "$OUT/moe_kernel_gate_android"

"$CXX" $CXXFLAGS -static-libstdc++ bench/moe_stream_gate.cpp $ENGINE $ENGINE_INCLUDES \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -Wl,--start-group $LIBS_ANDROID -Wl,--end-group -llog -lm \
  -o "$OUT/moe_stream_gate_android"

"$STRIP" "$OUT/llama_baseline_android" "$OUT/moe_kernel_gate_android" "$OUT/moe_stream_gate_android"
echo "built harnesses in $OUT"
