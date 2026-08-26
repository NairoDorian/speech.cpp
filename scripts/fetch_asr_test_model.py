#!/usr/bin/env python3
"""Fetch the pinned ASR models used by the end-to-end text gates.

Three models, one per gate, all the smallest validated weights whose arch is
compiled into the unified runtime:

- moonshine-tiny Q8_0 (34 MB) -> asr_e2e_wer_test, the offline corpus-WER
  gate. UsefulSensors/moonshine-tiny; 4.60% WER on full LibriSpeech
  test-clean. Arch: src/runtime/arch/moonshine.
- moonshine-streaming-tiny Q8_0 (48 MB) -> asr_stream_text_wer_test, the
  streaming-text corpus-WER gate. UsefulSensors/moonshine-streaming-tiny
  (upstream f8e9dfd); 4.52% offline / 4.54% streamed WER on full test-clean
  (transcribe.cpp's published parity run). Arch:
  src/runtime/arch/moonshine_streaming. HF repo revision 85ddff6,
  pinned 2026-08-20.
- whisper tiny.en (74 MB, legacy ggml .bin) -> asr_e2e_whisper_wer_test, the
  Whisper family gate. ggerganov/whisper.cpp ggml-tiny.en.bin - the canonical
  whisper.cpp distribution, MIT. Pinned 2026-08-26 because
  model_specs/whisper.json is catalog-only: its 16 packages point at
  Whisper-*-GGUF paths that do NOT exist in audio-cpp/audio.cpp-gguf, so there
  is no downloadable GGUF for the family. The legacy .bin loader
  (src/runtime/arch/whisper/bin_load.cpp) reads this format directly, which is
  why the gate uses it. Arch: src/runtime/arch/whisper.

The LibriSpeech fixtures the gates score against are NOT fetched here: the four
wav/txt pairs plus manifest.jsonl are vendored under
assets/asr_validation/librispeech/ and tracked in git. A pin for a
librispeech-test-clean-500w.tar.gz used to sit in this table; its HF dataset is
now gated (HTTP 401) and nothing referenced it, so a no-argument run of this
script failed for everyone. Removed 2026-08-26.

Each download is pinned by sha256 (the HF repo's LFS oid); a mismatched or
truncated download is deleted and reported, never installed. Destination is
the gitignored models/ directory — the gates are downloads, not vendored
assets, and their CTest registrations skip (exit 2) while a file is absent.

Usage:  uv run scripts/fetch_asr_test_model.py [--force] [--only NAME]
        (or any Python 3: python scripts/fetch_asr_test_model.py)
        NAME is a model filename substring, e.g. --only streaming

stdlib-only on purpose (urllib + hashlib): unlike fetch_silero_vad.py this
needs no torch/safetensors, so it runs on any Python without a venv.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True)
class PinnedModel:
    filename: str
    url: str
    sha256: str
    size: int


PINNED_MODELS = (
    PinnedModel(
        filename="moonshine-tiny-Q8_0.gguf",
        url=(
            "https://huggingface.co/handy-computer/moonshine-tiny-gguf/resolve/main/"
            "moonshine-tiny-Q8_0.gguf"
        ),
        sha256="2fd348d7b38f97d309cc3ec6848f3f57f537b80244950f07d2637e463f95a3a1",
        size=35_466_912,
    ),
    PinnedModel(
        filename="moonshine-streaming-tiny-Q8_0.gguf",
        url=(
            "https://huggingface.co/handy-computer/moonshine-streaming-tiny-gguf/resolve/main/"
            "moonshine-streaming-tiny-Q8_0.gguf"
        ),
        sha256="930e4622ad3a24158b91406c30c977fa6a26b34cb32d6ac3e57cfb23383a869e",
        size=50_462_816,
    ),
    PinnedModel(
        filename="ggml-tiny.en.bin",
        url=(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
            "ggml-tiny.en.bin"
        ),
        sha256="921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f",
        size=77_704_715,
    ),
    # Phase 10.5: the audio.cpp package for the canonical Qwen3-ASR engine
    # family (model_specs/qwen3_asr.json id qwen3_asr_0_6b_q8_0). Its
    # general.architecture is "audiocpp", so it exercises the C ABI's
    # framework routing as well as the engine gates.
    PinnedModel(
        filename="qwen3-asr-0.6b-q8_0.gguf",
        url=(
            "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/"
            "Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf"
        ),
        sha256="6c44ec2fb4cee513892d7863c1fcc3ea6b699ffa4d899b0ef4ab19956d9544f7",
        size=1_151_272_416,
    ),
    # Phase 10.5: Voxtral-Realtime. q4_k rather than the catalogue's default
    # q8_0 (5.1 GB) or bf16 (8.9 GB): this is a 4B family and the gate scores
    # transcription behaviour, not quantization. Same rule as the tiny models
    # above - pin the smallest package that exercises the family honestly.
    PinnedModel(
        filename="voxtral-mini-4b-realtime-2602-q4_k.gguf",
        url=(
            "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/"
            "Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q4_k.gguf"
        ),
        sha256="8cafef18ea3e4cad81da8ffc4e72b69d2eab2c159c2e68428e2e088accbfc7f8",
        size=3_097_662_432,
    ),
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(model: PinnedModel, force: bool) -> bool:
    out_path = MODELS_DIR / model.filename
    if out_path.exists() and not force:
        digest = sha256_of(out_path)
        if digest == model.sha256:
            print(f"ok: {out_path} already present and pinned ({model.size} bytes)")
            return True
        print(f"warning: {out_path} exists but sha256 {digest} != pinned {model.sha256}; "
              "re-downloading")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {model.url}")
    print(f"        to {out_path} ({model.size / 1e6:.1f} MB)")

    # Download to a sibling temp file and rename only after the pin verifies,
    # so an interrupted or corrupted transfer can never be mistaken for the
    # model by the test.
    with tempfile.NamedTemporaryFile(dir=out_path.parent, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(model.url, tmp_path)  # noqa: S310 - pinned https URL
        size = tmp_path.stat().st_size
        digest = sha256_of(tmp_path)
        if size != model.size or digest != model.sha256:
            print(f"error: download does not match pin (size {size} vs {model.size}, "
                  f"sha256 {digest} vs {model.sha256})", file=sys.stderr)
            tmp_path.unlink(missing_ok=True)
            return False
        tmp_path.replace(out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    print(f"ok: {out_path} ({model.size} bytes, sha256 {model.sha256})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if a pinned file is already present")
    parser.add_argument("--only", metavar="NAME", default="",
                        help="fetch only models whose filename contains NAME "
                             "(e.g. --only streaming)")
    args = parser.parse_args()

    selected = [m for m in PINNED_MODELS if args.only in m.filename]
    if not selected:
        names = ", ".join(m.filename for m in PINNED_MODELS)
        print(f"error: --only {args.only!r} matches none of: {names}", file=sys.stderr)
        return 1

    ok = True
    for model in selected:
        ok = fetch(model, args.force) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
