# Repository preparation

This checklist tracks the design handoff, not implementation of the engine.

- [x] Ground: inspect the workspace, recover device facts, and confirm GitHub access.
- [x] Sketch: compare three independent architecture candidates and review backend evidence.
- [x] Agree: synthesize the architecture and record tradeoffs. No implementation approval checkpoint was requested.
- [x] Implement: write the repository documentation, configuration examples, and DeepSeek handoff.
- [x] Verify: the scaffold check and staged whitespace check passed. GitHub readback confirmed a private, nonempty repository on main with the expected handoff tree.
- [x] Scrap: no unresolved conflicting ownership or memory contract remained after independent review.

The user selected the repository name `android-moe-engine` and private visibility.
The requested implementation work belongs to the later DeepSeek handoff.

## Host coding groundwork

Implemented the host build, checked extent primitive, help/version CLI, coding rules, and offline local/CI verifier. See [recorded validation](VALIDATION.md#coding-foundation-evidence-2026-10-19). Android toolchain/device and inference gates remain incomplete; continue with the remaining Phase 0 checks.
