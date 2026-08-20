#!/usr/bin/env python3
"""Fetch the pinned ASR models used by the end-to-end text gates.

Two models, one per gate, both MIT, both ported and WER-validated by
transcribe.cpp, both the smallest validated GGUF whose arch is compiled into
the unified runtime:

- moonshine-tiny Q8_0 (34 MB) -> asr_e2e_wer_test, the offline corpus-WER
  gate. UsefulSensors/moonshine-tiny; 4.60% WER on full LibriSpeech
  test-clean. Arch: src/runtime/arch/moonshine.
- moonshine-streaming-tiny Q8_0 (48 MB) -> asr_stream_text_wer_test, the
  streaming-text corpus-WER gate. UsefulSensors/moonshine-streaming-tiny
  (upstream f8e9dfd); 4.52% offline / 4.54% streamed WER on full test-clean
  (transcribe.cpp's published parity run). Arch:
  src/runtime/arch/moonshine_streaming. HF repo revision 85ddff6,
  pinned 2026-08-20.

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
