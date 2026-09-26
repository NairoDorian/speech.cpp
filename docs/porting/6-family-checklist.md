# Family Checklist

> **speech.cpp note (2026-09-26).** This guide comes from transcribe.cpp and is kept verbatim so parent diffs stay easy to triage. **In speech.cpp, translate it as follows** ([`PLAN.md`](../../PLAN.md) *North star*, rules R1–R11):
> - `src/arch/<family>/` means **an engine package**: `src/models/<family>/` (graphs, assets, thin session), `include/engine/models/<family>/` and `model_specs/<family>.json`, on the shared `framework/*` layers. **No private KV cache, decode loop, loader, mel or scheduler** (rule R5). Existing transcribe arches live in `src/runtime/arch/` only until they retire.
> - Before starting, answer the scope question (R9: does FreeSpeech need it, and in which profile?) and run the upstream-first check (R4 / L16).
> - Tests follow the quick/full tiers (one short clip by default; `AGENTS.md`). The port bar is WER equal to the reference baseline.
> - The family must build in the BUNDLE profile for desktop **and** Android (R1, R2).


Use this checklist to decide whether a model family is ready to call
supported.

## Research

- Family note exists at `docs/porting/families/<family>.md`.
- Canonical reference is identified.
- Instrumented reference is identified.
- Third-party implementations have been surveyed.
- Bridge validation is recorded if canonical and instrumented references
  differ.
- Hardware and package versions are recorded.

## Reference And Artifacts

- Reference run command is recorded.
- Reference transcript and token output are recorded when available.
- Golden manifest exists under `tests/golden/<family>/`.
- Reference dumps can be generated or restored.
- Output payload hashes are verified.
- Artifact cache behavior is documented.

## Conversion

- Accuracy GGUF conversion works.
- Converter manifest exists or converter prints all required fields.
- Tensor names and layouts match the reference dump conventions.
- Tokenizer metadata is correct.
- Frontend metadata is correct.
- Hparams and capability KVs are correct.
- Skipped, tied, fused, or derived tensors are documented.

## Loader And Tests

- `src/arch/<family>/` exists.
- Family is registered in `src/transcribe-arch.cpp`.
- Tiny synthetic fixture exists.
- `tests/<family>_smoke.cpp` passes in default CI when fixture
  generation is available.
- `tests/<family>_real_smoke.cpp` is gated behind
  `TRANSCRIBE_BUILD_REAL_MODEL_TESTS`.
- E2E public ABI smoke exists.
- Missing real model paths skip with return code `77`.

## Numerical Validation

- `tests/tolerances/<family>.json` exists.
- CPU accuracy GGUF validation passes.
- Primary accelerator validation passes or is documented unsupported.
- Quant CLI output-validity passes for each enabled quant; do not use tensor/numeric comparisons as a quant acceptance gate.
- Validation report is written under `reports/validate/<family>/`.

## Benchmarks

- Reference implementation benchmark exists.
- C++ CPU benchmark exists.
- Primary accelerator benchmark exists or is documented unsupported.
- Quant benchmark rows exist for enabled quants.
- Benchmark comparison works with configured thresholds.
- Transcript or token hash is included in benchmark reports.

## Promotion

- README supported-model list is updated.
- Known limitations are documented.
- Required setup commands are documented.
- Default `ctest` passes on a clean checkout without real model files.
- Optional real-model gate command is documented.
