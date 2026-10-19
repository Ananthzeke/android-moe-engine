#!/bin/sh
set -eu
: "${AME_TOOLS:?set AME_TOOLS}"
REPO=${AME_REPO:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
NDK="$AME_TOOLS/android-ndk-r27c"
LLAMA="$AME_TOOLS/src/llama.cpp"
HOST="$AME_TOOLS/build/llama-host"
ANDROID="$AME_TOOLS/build/llama-android"
OUT="$AME_TOOLS/build/harness"
CXX="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++"
STRIP="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"
ENGINE="src/backend/moe_stream.cpp src/backend/artifact.cpp src/common/sha256.cpp src/model/file_extent.cpp src/model/expert_layout.cpp src/runtime/prefill_plan.cpp src/residency/file_source.cpp src/residency/store.cpp"
INCLUDES="-Isrc/backend -Isrc/model -Isrc/residency -Isrc/common -Isrc/runtime -I$LLAMA/include -I$LLAMA/src -I$LLAMA/ggml/include -I$LLAMA/vendor/cpp-httplib -I$LLAMA/vendor/nlohmann"
mkdir -p "$OUT"
cd "$REPO"
g++ -std=c++20 -O2 -DNDEBUG -DAME_STREAMING=1 apps/server/main.cpp "$LLAMA/vendor/cpp-httplib/httplib.cpp" $ENGINE $INCLUDES   -Wl,--whole-archive "$HOST/src/libllama.a" -Wl,--no-whole-archive   -Wl,--start-group "$HOST/ggml/src/libggml.a" "$HOST/ggml/src/libggml-base.a" "$HOST/ggml/src/libggml-cpu.a" -Wl,--end-group   -lpthread -ldl -lm -o "$OUT/ame-server-host"
"$CXX" -std=c++20 -O2 -DNDEBUG -DAME_STREAMING=1 -static-libstdc++ apps/server/main.cpp "$LLAMA/vendor/cpp-httplib/httplib.cpp" $ENGINE $INCLUDES   -Wl,--whole-archive "$ANDROID/src/libllama.a" -Wl,--no-whole-archive   -Wl,--start-group "$ANDROID/ggml/src/libggml.a" "$ANDROID/ggml/src/libggml-base.a" "$ANDROID/ggml/src/libggml-cpu.a" -Wl,--end-group   -llog -lm -pthread -o "$OUT/ame-server-android"
"$STRIP" "$OUT/ame-server-android"
echo "built $OUT/ame-server-host and $OUT/ame-server-android"
