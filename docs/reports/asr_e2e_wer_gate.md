# End-to-end ASR WER gates (unified C ABI)

This report records the end-to-end speech-to-text validation of the unified
runtime: real GGUF ASR models loaded through `transcribe_open()`, the in-tree
LibriSpeech fixtures transcribed through `transcribe_run()` — and, since
2026-08-20, streamed through `transcribe_stream_begin/feed/finalize` — with
the text scored against the reference transcripts. Until these gates existed,
nothing in the merged tree ever checked *what the model wrote* — the bridge
tests validate plumbing (load, cursor algebra, segment folding) and pass
identically whether the transcript is correct or garbage.

Two gates, one pinned model each, one shared fetch script
(`uv run scripts/fetch_asr_test_model.py`):

| Gate | Path | Model | Validates |
|---|---|---|---|
| `asr_e2e_wer_test` | offline `transcribe_run` | moonshine-tiny Q8_0 (34 MB) | offline text |
| `asr_stream_text_wer_test` | streaming begin/feed/finalize + `transcribe_stream_get_text` | moonshine-streaming-tiny Q8_0 (48 MB) | streamed text, offline text, and their agreement |

## The offline gate (asr_e2e_wer_test)

`tests/asr_e2e_wer_test.cpp`, registered in CTest as `asr_e2e_wer_test`
whenever `SPEECHCPP_ENABLE_UNIFIED_ABI=ON` and
`SPEECHCPP_ENABLE_TRANSCRIBE_ARCHES=ON`. (The streaming gate below is
registered under the same conditions; both score text identically through
the shared `tests/asr_test_text.h`.)

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

## The streaming-text gate (asr_stream_text_wer_test, 2026-08-20)

Offline moonshine correctly reports `supports_streaming == false`, so the
offline gate cannot exercise streaming, and `abi_stream_hello` never reads
text. The follow-up this report used to end on — "a streaming-family GGUF
small enough to pin" — exists: transcribe.cpp ships
**moonshine-streaming-tiny Q8_0** (48 MB, MIT,
`huggingface.co/handy-computer/moonshine-streaming-tiny-gguf`, upstream
`UsefulSensors/moonshine-streaming-tiny` @ `f8e9dfd`), WER-validated on the
full test-clean split at **4.52% offline / 4.54% streamed**. Its arch is
in-tree (`src/runtime/arch/moonshine_streaming`).

- Pinned sha256 (= the repo's LFS oid, HF revision `85ddff6`, verified
  2026-08-20):
  `930e4622ad3a24158b91406c30c977fa6a26b34cb32d6ac3e57cfb23383a869e`
  (50,462,816 bytes).
- `tests/asr_stream_text_wer_test.cpp` runs every fixture through ONE session
  twice: offline `transcribe_run`, then a begin → odd-sized feeds (~100–400 ms,
  never aligned to an internal chunk) → finalize cycle, reading the final
  transcript from `transcribe_stream_get_text().full_text` (asserting the
  documented finalize contract: tentative text empty). Interleaving run and
  stream on one session also proves run/stream mode switching and
  `stream_reset` clear per-utterance text state.
- Gates: streamed corpus WER ≤ 10%, offline corpus WER ≤ 10%, and
  streamed-vs-offline corpus divergence ≤ 3 words (same weights — divergence
  is a streaming-path defect by construction; transcribe.cpp measured +8 word
  edits over ~52k words for this model). Skips (exit 2) while the model file
  is absent.

### Measured baseline (2026-08-20, same environment as above)

| Fixture | ref words | streamed edits | offline edits | stream vs offline |
|---|---:|---:|---:|---:|
| test_clean 6930-75918-0000 | 8 | 0 | 0 | 0 |
| test_clean 6930-75918-0001 | 43 | 1 | 1 | 0 |
| test_other 7902-96591-0000 | 9 | 2 | 2 | 0 |
| test_other 7902-96591-0001 | 9 | 0 | 0 | 0 |
| **corpus** | **69** | **3 (4.35%)** | **3 (4.35%)** | **0** |

The streamed transcript is **word-identical to the offline transcript** on
this corpus. Both differ from the references by `FORWARDED` → `VOTED` (the
same substitution offline moonshine-tiny makes) and `I AM` → `I'M` (a
contraction, two edits). Streamed decode ran at RTF 0.55 (~2x real time) on
CPU — slower than offline RTF 0.03 because a streaming session re-decodes
incrementally as audio arrives, which is the cost being bought for streaming
output.
