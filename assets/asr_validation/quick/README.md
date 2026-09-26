# Quick ASR fixture (1 clip)

One 3.5 s LibriSpeech test-clean clip (`6930-75918-0000`, CC BY 4.0), copied
from `../librispeech/`. It is the default fixture for every model-backed ASR
verdict / parity test: a first-pass check loads the model and transcribes one
short clip, so each test takes seconds, not minutes.

The 4-clip corpus in `../librispeech/` is used by the WER gates and by the
`*_full` verdict/parity variants, registered only with
`-DSPEECHCPP_FULL_ASR_TESTS=ON` (label `full`).
