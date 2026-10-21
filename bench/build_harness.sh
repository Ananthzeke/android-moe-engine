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
ENGINE="src/backend/moe_stream.cpp src/backend/artifact.cpp src/common/sha256.cpp src/model/file_extent.cpp src/model/expert_layout.cpp src/residency/file_source.cpp src/residency/store.cpp src/runtime/prefill_plan.cpp"
ENGINE_INCLUDES="-Isrc/backend -Isrc/model -Isrc/residency -Isrc/common -Isrc/runtime -I$LLAMA/src"
LIBS_HOST="$HOST_BUILD/ggml/src/libggml.a $HOST_BUILD/ggml/src/libggml-base.a $HOST_BUILD/ggml/src/libggml-cpu.a"
LIBS_ANDROID="$ANDROID_BUILD/ggml/src/libggml.a $ANDROID_BUILD/ggml/src/libggml-base.a $ANDROID_BUILD/ggml/src/libggml-cpu.a"

mkdir -p "$OUT"
cd "$AME_REPO"

g++ $CXXFLAGS bench/llama_baseline.cpp $ENGINE $ENGINE_INCLUDES \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -Wl,--whole-archive "$HOST_BUILD/src/libllama.a" -Wl,--no-whole-archive -Wl,--start-group $LIBS_HOST -Wl,--end-group \
  -lpthread -ldl -lm -o "$OUT/llama_baseline_host"

g++ $CXXFLAGS bench/moe_kernel_gate.cpp \
  src/model/file_extent.cpp src/model/expert_layout.cpp \
  src/residency/file_source.cpp src/residency/store.cpp \
  -Isrc/model -Isrc/residency -Isrc/common \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -Wl,--start-group $LIBS_HOST -Wl,--end-group \
  -lpthread -ldl -lm -o "$OUT/moe_kernel_gate_host"

g++ $CXXFLAGS bench/moe_stream_gate.cpp $ENGINE $ENGINE_INCLUDES \
  -I"$LLAMA/include" -I"$LLAMA/src" -I"$LLAMA/ggml/include" \
  -Wl,--whole-archive "$HOST_BUILD/src/libllama.a" -Wl,--no-whole-archive \
  -Wl,--start-group $LIBS_HOST -Wl,--end-group \
  -lpthread -ldl -lm -o "$OUT/moe_stream_gate_host"

"$CXX" $CXXFLAGS -march=armv8-a+crypto -Isrc/common -c src/common/sha256_arm.cpp -o "$OUT/sha256_arm.o"

"$CXX" $CXXFLAGS -static-libstdc++ bench/llama_baseline.cpp $ENGINE $ENGINE_INCLUDES \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -Wl,--whole-archive "$ANDROID_BUILD/src/libllama.a" -Wl,--no-whole-archive -Wl,--start-group $LIBS_ANDROID -Wl,--end-group \
  "$OUT/sha256_arm.o" -llog -lm -o "$OUT/llama_baseline_android"

"$CXX" $CXXFLAGS -static-libstdc++ bench/moe_kernel_gate.cpp \
  src/model/file_extent.cpp src/model/expert_layout.cpp \
  src/residency/file_source.cpp src/residency/store.cpp \
  -Isrc/model -Isrc/residency -Isrc/common \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -Wl,--start-group $LIBS_ANDROID -Wl,--end-group -llog -lm \
  -o "$OUT/moe_kernel_gate_android"

"$CXX" $CXXFLAGS -static-libstdc++ bench/moe_stream_gate.cpp $ENGINE $ENGINE_INCLUDES \
  -I"$LLAMA/include" -I"$LLAMA/src" -I"$LLAMA/ggml/include" \
  -Wl,--whole-archive "$ANDROID_BUILD/src/libllama.a" -Wl,--no-whole-archive \
  -Wl,--start-group $LIBS_ANDROID -Wl,--end-group "$OUT/sha256_arm.o" -llog -lm \
  -o "$OUT/moe_stream_gate_android"

"$CXX" $CXXFLAGS -static-libstdc++ bench/moe_quantized_gate.cpp   src/model/file_extent.cpp src/model/expert_layout.cpp   src/residency/file_source.cpp src/residency/store.cpp   -Isrc/model -Isrc/residency -Isrc/common   -I"$LLAMA/include" -I"$LLAMA/ggml/include"   -Wl,--start-group $LIBS_ANDROID -Wl,--end-group -llog -lm   -o "$OUT/moe_quantized_gate_android"

"$CXX" $CXXFLAGS -static-libstdc++ -DAME_STREAMING=1 bench/complete_runner.cpp $ENGINE $ENGINE_INCLUDES \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -Wl,--whole-archive "$ANDROID_BUILD/src/libllama.a" -Wl,--no-whole-archive \
  -Wl,--start-group $LIBS_ANDROID -Wl,--end-group "$OUT/sha256_arm.o" -llog -lm \
  -o "$OUT/complete_runner_android"

"$STRIP" "$OUT/llama_baseline_android" "$OUT/moe_kernel_gate_android" "$OUT/moe_stream_gate_android" "$OUT/moe_quantized_gate_android" "$OUT/complete_runner_android"
echo "built harnesses in $OUT"