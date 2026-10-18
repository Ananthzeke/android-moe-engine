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

For a single development build, use the venv tool paths:

```sh
.venv/bin/cmake --preset host-debug
.venv/bin/cmake --build --preset host-debug
.venv/bin/ctest --preset host-debug
./build/host-debug/ame --help
```

Read [coding rules](docs/CODING.md), [architecture](docs/ARCHITECTURE.md), and [validation](docs/VALIDATION.md). Submit reviewable units with commands, actual results, and limitations. Keep weights and private logs untracked. License selection remains an owner decision.

`android-arm64` is conditional on `ANDROID_NDK_HOME`; it targets ARM64/API31 with tests disabled and never substitutes a host toolchain. Set that variable to an installed NDK and use the matching configure/build presets. No NDK identity is pinned yet: the implementer must inspect and lock the actual NDK before claiming reproducibility. This preset's cross-build, ELF inspection, and physical-device behavior are unverified. Host checks do not complete Phase 0.
