#!/bin/sh
set -eu
: "${AME_TOOLS:?set AME_TOOLS}"
REPO=${AME_REPO:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
NDK="$AME_TOOLS/android-ndk-r27c"
LLAMA="$AME_TOOLS/src/llama.cpp"
HOST="$AME_TOOLS/build/llama-host"
ANDROID="$AME_TOOLS/build/llama-android"
OUT="$AME_TOOLS/build/harness"
CMAKE=${AME_CMAKE:-$REPO/.venv/bin/cmake}
PYTHON=${AME_PYTHON:-$REPO/.venv/bin/python}
CXX="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++"
STRIP="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"
ENGINE="src/backend/moe_stream.cpp src/backend/artifact.cpp src/common/sha256.cpp src/model/file_extent.cpp src/model/expert_layout.cpp src/runtime/prefill_plan.cpp src/residency/file_source.cpp src/residency/store.cpp"
INCLUDES="-Isrc/backend -Isrc/model -Isrc/residency -Isrc/common -Isrc/runtime -I$LLAMA/include -I$LLAMA/src -I$LLAMA/ggml/include -I$LLAMA/common -I$LLAMA/vendor/cpp-httplib -I$LLAMA/vendor/nlohmann"
mkdir -p "$OUT"
cd "$REPO"
"$PYTHON" - "$LLAMA" configs/dependencies.lock.json <<'PY'
import json, subprocess, sys
expected = next(x['commit'] for x in json.load(open(sys.argv[2]))['backends']
                if x['id'] == 'llama.cpp')
actual = subprocess.check_output(['git', '-C', sys.argv[1], 'rev-parse', 'HEAD'], text=True).strip()
if actual != expected:
    raise SystemExit('backend revision does not match dependency lock')
PY
for BUILD in "$HOST" "$ANDROID"; do
  "$CMAKE" -S "$LLAMA" -B "$BUILD" -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_BORINGSSL=OFF -DLLAMA_BUILD_LIBRESSL=OFF
  "$CMAKE" --build "$BUILD" --target llama-common --parallel "${AME_BUILD_JOBS:-2}"
done
HOST_LIBS="$HOST/common/libllama-common.a $HOST/common/libllama-common-base.a $HOST/vendor/cpp-httplib/libcpp-httplib.a $HOST/ggml/src/libggml.a $HOST/ggml/src/libggml-base.a $HOST/ggml/src/libggml-cpu.a"
ANDROID_LIBS="$ANDROID/common/libllama-common.a $ANDROID/common/libllama-common-base.a $ANDROID/vendor/cpp-httplib/libcpp-httplib.a $ANDROID/ggml/src/libggml.a $ANDROID/ggml/src/libggml-base.a $ANDROID/ggml/src/libggml-cpu.a"
g++ -std=c++20 -O2 -DNDEBUG tests/chat_template_test.cpp $INCLUDES \
  -Wl,--whole-archive "$HOST/src/libllama.a" -Wl,--no-whole-archive \
  -Wl,--start-group $HOST_LIBS -Wl,--end-group -lpthread -ldl -lm -o "$OUT/chat-template-test"
"$OUT/chat-template-test" tests/fixtures/qwen36-chat-template.jinja
g++ -std=c++20 -O2 -DNDEBUG -DAME_STREAMING=1 apps/server/main.cpp $ENGINE $INCLUDES \
  -Wl,--whole-archive "$HOST/src/libllama.a" -Wl,--no-whole-archive \
  -Wl,--start-group $HOST_LIBS -Wl,--end-group -lpthread -ldl -lm -o "$OUT/ame-server-host"
"$CXX" -std=c++20 -O2 -march=armv8-a+crypto -Isrc/common -c src/common/sha256_arm.cpp -o "$OUT/sha256_arm.o"
"$CXX" -std=c++20 -O2 -DNDEBUG -DAME_STREAMING=1 -static-libstdc++ apps/server/main.cpp $ENGINE $INCLUDES \
  -Wl,--whole-archive "$ANDROID/src/libllama.a" -Wl,--no-whole-archive \
  -Wl,--start-group $ANDROID_LIBS -Wl,--end-group "$OUT/sha256_arm.o" -llog -lm -pthread -o "$OUT/ame-server-android"
"$STRIP" "$OUT/ame-server-android"
echo "built $OUT/ame-server-host and $OUT/ame-server-android"