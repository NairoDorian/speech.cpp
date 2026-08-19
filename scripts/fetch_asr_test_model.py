#!/usr/bin/env python3
"""Fetch the pinned ASR model used by the end-to-end WER gate (asr_e2e_wer_test).

Model: moonshine-tiny Q8_0 (UsefulSensors/moonshine-tiny, ported and
WER-validated by transcribe.cpp; 4.60% WER on full LibriSpeech test-clean).
Chosen for the gate because it is the smallest WER-validated GGUF whose arch
is compiled into the unified runtime (src/runtime/arch/moonshine), so a 34 MB
download turns asr_e2e_wer_test from a permanent skip into a real gate.

Source: https://huggingface.co/handy-computer/moonshine-tiny-gguf (MIT).
The download is pinned by sha256 (the repo's LFS oid, verified 2026-08-20);
a mismatched or truncated download is deleted and reported, never installed.

Destination: models/moonshine-tiny-Q8_0.gguf — models/ is gitignored; the
test and CMake registration both point at this canonical path.

Usage:  uv run scripts/fetch_asr_test_model.py [--force]
        (or any Python 3: python scripts/fetch_asr_test_model.py)

stdlib-only on purpose (urllib + hashlib): unlike fetch_silero_vad.py this
needs no torch/safetensors, so it runs on any Python without a venv.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://huggingface.co/handy-computer/moonshine-tiny-gguf/resolve/main/"
    "moonshine-tiny-Q8_0.gguf"
)
PINNED_SHA256 = "2fd348d7b38f97d309cc3ec6848f3f57f537b80244950f07d2637e463f95a3a1"
PINNED_SIZE = 35_466_912

OUT_PATH = Path(__file__).resolve().parents[1] / "models" / "moonshine-tiny-Q8_0.gguf"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the pinned file is already present")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        digest = sha256_of(OUT_PATH)
        if digest == PINNED_SHA256:
            print(f"ok: {OUT_PATH} already present and pinned ({PINNED_SIZE} bytes)")
            return 0
        print(f"warning: {OUT_PATH} exists but sha256 {digest} != pinned {PINNED_SHA256}; "
              "re-downloading")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL}")
    print(f"        to {OUT_PATH} ({PINNED_SIZE / 1e6:.1f} MB)")

    # Download to a sibling temp file and rename only after the pin verifies,
    # so an interrupted or corrupted transfer can never be mistaken for the
    # model by the test.
    with tempfile.NamedTemporaryFile(dir=OUT_PATH.parent, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)  # noqa: S310 - pinned https URL
        size = tmp_path.stat().st_size
        digest = sha256_of(tmp_path)
        if size != PINNED_SIZE or digest != PINNED_SHA256:
            print(f"error: download does not match pin (size {size} vs {PINNED_SIZE}, "
                  f"sha256 {digest} vs {PINNED_SHA256})", file=sys.stderr)
            tmp_path.unlink(missing_ok=True)
            return 1
        tmp_path.replace(OUT_PATH)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    print(f"ok: {OUT_PATH} ({PINNED_SIZE} bytes, sha256 {PINNED_SHA256})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
