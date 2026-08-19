# End-to-end ASR WER gate (unified C ABI)

This report records the first end-to-end speech-to-text validation of the
unified runtime: a real GGUF ASR model loaded through `transcribe_open()`,
the in-tree LibriSpeech fixtures transcribed through `transcribe_run()`, and
the text scored against the reference transcripts. Until this gate existed,
nothing in the merged tree ever checked *what the model wrote* — the bridge
tests validate plumbing (load, cursor algebra, segment folding) and pass
identically whether the transcript is correct or garbage.

## The gate

`tests/asr_e2e_wer_test.cpp`, registered in CTest as `asr_e2e_wer_test`
whenever `SPEECHCPP_ENABLE_UNIFIED_ABI=ON` and
`SPEECHCPP_ENABLE_TRANSCRIBE_ARCHES=ON`.

- Links ONLY the public C ABI (`transcribe.dll`) — no engine_runtime, no ggml —
  so a WER regression observed here is a regression in exactly what a language
  binding ships.
- Scans `assets/asr_validation/librispeech/` for `.wav`/`.txt` pairs (four are
  in-tree: two test-clean, two test-other), transcribes each, and scores
  **corpus WER** — total word edits over total reference words — with
  LibriSpeech-style normalization (uppercase, `[A-Z0-9']` kept, typographic
  apostrophes folded, everything else a separator).
- The gate is corpus WER ≤ 10% by default (`SPEECHCPP_ASR_E2E_MAX_WER`
  overrides). It also fails if any hypothesis is empty.
- The model is **not in-tree** (`models/` is gitignored). The test skips
  (exit 2, `SKIP_RETURN_CODE`) while the model file is absent, so checkouts
  that never fetch it stay green. Fetch with:

  ```bash
  uv run scripts/fetch_asr_test_model.py   # or any Python 3; stdlib only
  ```

## The model

`moonshine-tiny-Q8_0.gguf` — Useful Sensors' 27M-parameter moonshine-tiny as
ported and WER-validated by transcribe.cpp (4.60% WER on the full LibriSpeech
test-clean split for this exact quant; see transcribe.cpp
`docs/models/moonshine-tiny.md`). Chosen because it is the smallest
WER-validated GGUF whose arch is compiled into the unified runtime
(`src/runtime/arch/moonshine`), making the gate a 34 MB download.

- Source: `huggingface.co/handy-computer/moonshine-tiny-gguf` (MIT)
- Pinned sha256 (= the repo's LFS oid, verified 2026-08-20):
  `2fd348d7b38f97d309cc3ec6848f3f57f537b80244950f07d2637e463f95a3a1`
  (35,466,912 bytes). `scripts/fetch_asr_test_model.py` refuses to install a
  download that does not match the pin.

## Measured baseline (2026-08-20)

Environment: Windows 11 Pro 10.0.26200, Intel i9-13900H, 31.6 GiB RAM.
Build: Ninja Release, clang-cl, CPU backend, `AUDIOCPP_MODEL_SET=full`,
`SPEECHCPP_ENABLE_UNIFIED_ABI=ON`, `SPEECHCPP_ENABLE_TRANSCRIBE_ARCHES=ON`,
OpenMP OFF.

| Fixture | ref words | edits | WER |
|---|---:|---:|---:|
| test_clean 6930-75918-0000 | 8 | 0 | 0% |
| test_clean 6930-75918-0001 | 43 | 1 | 2.33% |
| test_other 7902-96591-0000 | 9 | 0 | 0% |
| test_other 7902-96591-0001 | 9 | 0 | 0% |
| **corpus** | **69** | **1** | **1.45%** |

The single edit is `FORWARDED` → `VOTED` in the long test-clean utterance — a
plausible tiny-model substitution, consistent with moonshine-tiny's published
4.6% test-clean WER. Aggregate decode ran at RTF 0.033 (~30x real time) on CPU.

## Why the bound is 10%

One word on this corpus costs 1.45 pp, so the bound cannot be meaningfully
tighter than a few words. 10% allows 6 edits: far above single-word jitter
from a backend or quant change, far below the 50–100% WER that structural
breakage (wrong mel layout, desynchronized decoder, tokenizer drift) produces.
The gate is a tripwire for "transcription broke", not a WER benchmark — the
full-split benchmark numbers live with each family's Stage 7 validation (plan
§3).

## Streaming

`abi_stream_hello` given this model skips: offline moonshine correctly reports
`supports_streaming == false` (the streaming family is `moonshine_streaming`,
a separate arch). The streaming surface remains validated against `silero_vad`
segments; a streaming-family GGUF small enough to pin is the natural follow-up
once one is published.
