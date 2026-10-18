# Contributing

Use Python 3.10+, GCC or Clang with C++20 support, and the pinned development tools:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python tools/verify.py
```

Only bootstrap requires network access. Verification downloads nothing and fails if tooling is missing. It checks pinned clang-format 18.1.8, documentation/configuration, then configures, builds, and runs CTest for `host-debug`, `host-release`, and `host-sanitizers` (ASan + UBSan). CI runs this same full matrix through the same command. Tool lookup prefers binaries beside the invoking Python, then PATH. CMake requires 3.22+; the bootstrap pins 3.31.6 and Ninja 1.11.1.4. Compiler and OS identity still need recording for experimental results.

Fix C++ formatting separately:

```sh
.venv/bin/python tools/verify.py --format
```

For a single development build, activate the venv so CMake can discover Ninja:

```sh
. .venv/bin/activate
cmake --preset host-debug
cmake --build --preset host-debug
ctest --preset host-debug
./build/host-debug/ame --help
```

Read [coding rules](docs/CODING.md), [architecture](docs/ARCHITECTURE.md), and [validation](docs/VALIDATION.md). Submit reviewable units with commands, actual results, and limitations. Keep weights and private logs untracked. License selection remains an owner decision.

`android-arm64` is conditional on `ANDROID_NDK_HOME`; it targets ARM64/API31 with tests disabled and never substitutes a host toolchain. Set that variable to the NDK pinned in [the dependency lock](configs/dependencies.lock.json) (r27c, 27.2.12479018), then configure and build with the matching presets. The cross-build, AArch64 ELF inspection, and physical-device probe execution have passed; backend builds and inference remain unverified. Host checks alone do not establish target inference.