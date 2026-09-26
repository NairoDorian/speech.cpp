# CrispASR triage ledger (reference parent #3)

> **Status:** binding (decision V7-D2, 2026-09-26; see [`../LEDGER.md`](../LEDGER.md) and [`../../PLAN.md`](../../PLAN.md) §6).
> **Checkout:** `../CrispASR` (sibling of `audio.cpp/`, `transcribe.cpp/`), origin `https://github.com/CrispStrobe/CrispASR`
> **Relationship:** a reference parent, **not** a git parent. Nothing is merged. Families are ported into engine packages (rule C-R1), with PyTorch as the oracle (C-R2).

watermark: dfcdacae (2026-09-26, v0.8.37)

## How to triage

- **Triage by family, on demand.** Do it when porting a family, refreshing an overlapping family, or doing the ggml patch audit. Never triage commit by commit: the repo averages ~40 commits a day.
- To see what changed in a family since the watermark:
  `git -C ../CrispASR log --oneline <watermark>..HEAD -- src/<model>.cpp src/<model>.h examples/cli/crispasr_backend_<model>.cpp models/convert-<model>*.py tools/reference_backends/<model>*`
- Record one row per family touched. Advance the watermark only when every family row below is dispositioned at the new sha.

## Ledger

| Family / item | CrispASR files | Disposition | speech.cpp target | Date |
|---|---|---|---|---|
| ggml fork patches 1–6, #10, #15 | `ggml` submodule `2f5a80d2`, UPSTREAM.md | PENDING — audit during the ggml ≥ 0.25.3 bump (C0.3) | `patches/` | – |
| stage diff harness | `tools/dump_reference.py`, `tools/reference_backends/*`, `examples/crispasr-diff` | PENDING (C0.1) | `tools/`, `tests/` | – |
| TTS → ASR round-trip gate | `.github/workflows/regression.yml` + manifest | PENDING (C0.2) | `tests/` | – |
