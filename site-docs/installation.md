# Installation from a clean Linux host

These commands target Linux x86-64 and produce a host CLI plus an Android
ARM64 server. They were checked against the locked tool versions; commands not
physically re-run here are marked.

## Prerequisites

- Python 3.10+
- C++20 compiler, GNU Make/Ninja, git, curl
- CMake 3.22+ (locked development version: 3.31.6)
- Android NDK **r27c**, `27.2.12479018`, API 31, arm64-v8a
- About 25 GB free for checkouts/builds plus model storage
- A GitHub checkout of this repository

The authoritative pins are in
[`configs/dependencies.lock.json`](https://github.com/Ananthzeke/android-moe-engine/blob/main/configs/dependencies.lock.json)
and [`requirements-dev.txt`](https://github.com/Ananthzeke/android-moe-engine/blob/main/requirements-dev.txt).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
```

## Required AME_TOOLS layout

Both inference build scripts require the same root:

```text
$AME_TOOLS/
  android-ndk-r27c/
  src/llama.cpp/
  build/llama-host/
  build/llama-android/
  build/harness/          # created by scripts
  models/                 # weights, never committed
```

```sh
export AME_TOOLS=/absolute/path/to/ame-tools
export AME_LLAMA_SOURCE_DIR=$AME_TOOLS/src/llama.cpp
export AME_LLAMA_HOST_LIB_DIR=$AME_TOOLS/build/llama-host
```

## Foundation CLI (no llama.cpp required)

This is the offline project verifier and CLI foundation, separate from the
inference-server build.

```sh
. .venv/bin/activate
cmake --preset host-debug
cmake --build --preset host-debug
ctest --preset host-debug
./build/host-debug/ame --help
```

Run the full offline gate:

```sh
.venv/bin/python tools/verify.py
# Debug, Release, ASan+UBSan: 12/12 tests in the current tree
```

## Pinned backend

```sh
mkdir -p "$AME_TOOLS/src"
git clone https://github.com/ggml-org/llama.cpp.git "$AME_TOOLS/src/llama.cpp"
git -C "$AME_TOOLS/src/llama.cpp" checkout ff5a23376725e5a889fa0bfb987b7541dc304da0
git -C "$AME_TOOLS/src/llama.cpp" apply "$PWD/patches/0001-ggml-moe-materializer.patch"
git -C "$AME_TOOLS/src/llama.cpp" apply "$PWD/patches/0002-llama-no-eager-mmap-prefetch.patch"
git -C "$AME_TOOLS/src/llama.cpp" add --intent-to-add -- ggml/include/ggml-moe.h
```

The scripts validate the backend revision against the dependency lock. Do not
change only one of the host/Android libraries; rebuild both after a patch
change.

Build the backend libraries:

```sh
.venv/bin/cmake -S "$AME_TOOLS/src/llama.cpp" -B "$AME_TOOLS/build/llama-host" \
  -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF
.venv/bin/cmake --build "$AME_TOOLS/build/llama-host" --target llama -j 2
```

For Android, configure the same pinned source with the NDK r27c toolchain for
arm64-v8a/API 31 into `$AME_TOOLS/build/llama-android`, then build the
`llama`, `llama-common`, and `ggml` targets. The repository's
`android-arm64` preset builds the project foundation; it does not replace the
pinned llama.cpp Android libraries. This configure step was not re-executed on
this documentation branch.

## Build the harnesses and server

```sh
AME_TOOLS="$AME_TOOLS" sh bench/build_harness.sh
AME_TOOLS="$AME_TOOLS" sh bench/build_server.sh
```

`build_harness.sh` now compiles `src/common/sha256_arm.cpp` with
`-march=armv8-a+crypto` and links `sha256_arm.o` into every Android harness
that includes `sha256.cpp`. This was a genuine link blocker: the runtime
dispatch calls ARM SHA2, while the scalar object is compiled without crypto.
The fix was built successfully in this branch.

`build_server.sh` likewise links `sha256_arm.o` into
`ame-server-android` and produces `ame-server-host`.

## Inspect the Android ELF

```sh
.venv/bin/python tools/check_elf.py "$AME_TOOLS/build/harness/ame-server-android"
file "$AME_TOOLS/build/harness/ame-server-android"
llvm-readelf -d "$AME_TOOLS/build/harness/ame-server-android" | grep NEEDED
```

The binary must be AArch64 and must not link a host-only library. The target
phone used here is ARMv8-A with `sha2`; unsupported CPUs fall back to scalar
SHA-256. This fallback is implemented but was not executed on a non-SHA2 phone.
