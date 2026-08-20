# ASR Validation Assets

Small, public ASR/STT fixtures for smoke validation. These are not training data
and are intentionally tiny.

## LibriSpeech

Path: `assets/asr_validation/librispeech/`

The LibriSpeech examples are taken from the `openslr/librispeech_asr` Hugging Face
dataset mirror, with upstream source OpenSLR SLR12. LibriSpeech is distributed
under CC BY 4.0.

Each example includes:

- a 16 kHz mono WAV file
- a `.txt` reference transcript
- one row in `manifest.jsonl`

The manifest rows include source metadata, split, speaker/chapter ids, duration,
audio path, transcript path, and the reference text.

## Consumers

- `asr_e2e_wer_test` transcribes every `.wav`/`.txt` pair here through the
  public C ABI and gates corpus WER against the references (see
  `docs/reports/asr_e2e_wer_gate.md`; models fetched by
  `scripts/fetch_asr_test_model.py`).
- `asr_stream_text_wer_test` streams the same pairs through the C ABI
  streaming surface and gates the streamed text the same way (same report
  and fetch script; second pinned model).
- `abi_stream_hello` uses the first test-clean WAV as its streaming input.
