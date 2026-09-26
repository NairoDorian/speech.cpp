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
  Whisper legacy-format gate. ggerganov/whisper.cpp ggml-tiny.en.bin - the
  canonical whisper.cpp distribution, MIT. Pinned 2026-08-26, when the family
  was believed to have no downloadable GGUF because model_specs/whisper.json
  points at Whisper-*-GGUF paths that do not exist in audio-cpp/audio.cpp-gguf.
  That was only half the picture (corrected 2026-09-23): parent transcribe.cpp
  publishes every Whisper variant as a GGUF under handy-computer/, pinned
  below. The .bin stays pinned because it gates the legacy loader.
- whisper tiny.en / tiny (46 MB each, transcribe.cpp GGUF, Q8_0) -> the Whisper
  GGUF gates (English and multilingual). handy-computer/whisper-*-gguf.

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
import os
import shutil
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
    # Phase 11 W2b: the family's real distribution format. transcribe.cpp
    # publishes every Whisper variant as a GGUF under handy-computer/ (see
    # ../transcribe.cpp/catalog/whisper-*.json, "published_repo"); the 2026-08-26
    # note below that "no downloadable GGUF exists" only ever held for
    # audio-cpp/audio.cpp-gguf. tiny.en exercises the English prefix; the
    # multilingual tiny exercises language detection / <|lang|> / translate.
    # URLs pin the repo commit, not main.
    PinnedModel(
        filename="whisper-tiny.en-Q8_0.gguf",
        url=(
            "https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/"
            "f2406b60206a289ed322f1cf8c492c070c042b1f/whisper-tiny.en-Q8_0.gguf"
        ),
        sha256="e8c9b73c06344307d8b346e07fbe93dd88d894627854bcff31523f1ce44394fa",
        size=45_904_544,
    ),
    PinnedModel(
        filename="whisper-tiny-Q8_0.gguf",
        url=(
            "https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/"
            "2678cc66038359b97c8e6fd6454c56fc9006d571/whisper-tiny-Q8_0.gguf"
        ),
        sha256="325b9c7997cd1eff81ef709d55766565e71be696130cc3a3d444713798706834",
        size=45_981_088,
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
    # The multilingual legacy .bin: the language-detection path of
    # whisper_bin_e2e_smoke and the GGUF-vs-.bin vocabulary check of
    # whisper_bin_tokenize_parity (both need a multilingual tiny).
    PinnedModel(
        filename="ggml-tiny.bin",
        url=(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/"
            "5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.bin"
        ),
        sha256="be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
        size=77_691_713,
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
    # The smallest published parakeet (FastConformer TDT+CTC, 114M params), in
    # the transcribe.cpp layout its canonical C-ABI arch reads. Gates the
    # shared src/runtime/conformer/ blocks - the S3 conformer memory chain from
    # transcribe.cpp (docs/upstream/transcribe_cpp_triage.md) had no local model
    # until this pin.
    PinnedModel(
        filename="parakeet-tdt_ctc-110m-Q8_0.gguf",
        url=(
            "https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/"
            "766f172fe70eb66785e3371664f53762e0fbafaa/parakeet-tdt_ctc-110m-Q8_0.gguf"
        ),
        sha256="7dd44c74a331d788a4e5f8b16913b3feb29ced22cf5613aad0e0f6cd30516296",
        size=135_373_280,
    ),
    # parakeet-unified-en-0.6b: the only parakeet on the buffered streaming
    # path (chunked_limited_with_rc), which transcribe.cpp 63baefe6's
    # tail-loss fix and its parakeet_buffered_stream_eos_smoke exercise. Q4_K_M,
    # the smallest published quant - the test checks the tail, not accuracy.
    PinnedModel(
        filename="parakeet-unified-en-0.6b-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/"
            "d5249700b2382bf5c5024c2421d101b8db54a629/parakeet-unified-en-0.6b-Q4_K_M.gguf"
        ),
        sha256="a8bf3de2b393bd14ead5a858c3748d5e3b07a20fdeabdd3b498fba4f463fa929",
        size=477_274_496,
    ),
    # Cohere Transcribe (03-2026) in BOTH layouts, for the arch-vs-engine verdict
    # (Phase 11b re-plan): the transcribe.cpp GGUF the `cohere` arch reads, and
    # the audio.cpp package the engine `cohere_asr` family reads. Same quant
    # (Q8_0) on both sides so the comparison is about the implementations.
    PinnedModel(
        filename="cohere-transcribe-03-2026-Q8_0.gguf",
        url=(
            "https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/"
            "ab667acedcb5d8d56837ef3ed3568c3ad50dde47/cohere-transcribe-03-2026-Q8_0.gguf"
        ),
        sha256="931916663432fd895423a4291a8400221802b288967ca2d435fc5e3141c9e71e",
        size=2_410_655_232,
    ),
    PinnedModel(
        filename="cohere-transcribe-03-2026-q8_0.audiocpp.gguf",
        url=(
            "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/"
            "Cohere-Transcribe-GGUF/cohere-transcribe-03-2026-q8_0.gguf"
        ),
        sha256="f3da2899fe5d918b3aefb0d3c0d4c963ca7171c42cb98439000d1228b224d51c",
        size=2_438_863_008,
    ),
    # Phase 11b engine-vs-arch parity (2026-09-24): the transcribe.cpp-layout
    # GGUFs the new / dual-layout engine packages must read. Smallest useful
    # quant per family; revisions and hashes from the HF API at pin time.
    PinnedModel(
        filename="canary-180m-flash-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/"
            "456e6049062ecf06f1a0f4607f2ee3dc80ebbf8a/canary-180m-flash-Q4_K_M.gguf"
        ),
        sha256="c8ae5758d7d4dc59c48d816474a33544e362ca52c635f96ee0b4b95e1b48f90c",
        size=139_223_744,
    ),
    PinnedModel(
        filename="MOSS-Transcribe-Diarize-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/"
            "bfa3d24438711391d8713c6ab0efd6264527757c/MOSS-Transcribe-Diarize-Q4_K_M.gguf"
        ),
        sha256="e371990f2c8f21ded18e18cfac8906b059b5ee7cbce154ed4d3c582c05100b8b",
        size=617_345_184,
    ),
    # Gated on HF (Health AI Developer Foundations terms): the download needs
    # an account that accepted them (HF_TOKEN).
    PinnedModel(
        filename="medasr-Q8_0.gguf",
        url=(
            "https://huggingface.co/handy-computer/medasr-gguf/resolve/"
            "4d7958fe23a9b8046951497f44d4aa67a9883cd3/medasr-Q8_0.gguf"
        ),
        sha256="5a391a2154416b96241b829ac2c7eed8b64d197f00372512c9bbf63f4849c978",
        size=127_712_448,
    ),
    PinnedModel(
        filename="canary-qwen-2.5b-Q8_0.gguf",
        url=(
            "https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/"
            "df576c1641eb59bb66bc3c396bbdd8b0b113825a/canary-qwen-2.5b-Q8_0.gguf"
        ),
        sha256="d89aad1285d5bd5aa441c464d3a4cf37bd5474f70705408e71558d8627415b34",
        size=2_797_548_928,
    ),
    PinnedModel(
        filename="gigaam-v3-e2e-rnnt-Q8_0.gguf",
        url=(
            "https://huggingface.co/handy-computer/gigaam-v3-e2e-rnnt-gguf/resolve/"
            "b9b68a835993df09018237a11d2a9b8c1925844d/gigaam-v3-e2e-rnnt-Q8_0.gguf"
        ),
        sha256="78d63b47723b7f8d78c6113a6ef983b5a86e2a86f6c273e1f5cb6967b1c4467a",
        size=273_724_832,
    ),
    PinnedModel(
        filename="gigaam-v3-ctc-Q8_0.gguf",
        url=(
            "https://huggingface.co/handy-computer/gigaam-v3-ctc-gguf/resolve/"
            "696b1bc14be5a4c423090bdc31da27793def4065/gigaam-v3-ctc-Q8_0.gguf"
        ),
        sha256="71e5c82890e9e243a6bd7575f129f5d1bd2c3ca3ae79aab75cfb1a6934c6a62b",
        size=271_803_328,
    ),
    PinnedModel(
        filename="Voxtral-Mini-3B-2507-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-gguf/resolve/"
            "b9e655d7a09a24394edf457e08651163ad22732d/Voxtral-Mini-3B-2507-Q4_K_M.gguf"
        ),
        sha256="3a6717aa8f8989108d260cbd237584289eec43cc987e10133c33643515936205",
        size=2_984_721_056,
    ),
    PinnedModel(
        filename="granite-4.0-1b-speech-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/"
            "9f243e662e448e8bcc64c3b1195413e106002334/granite-4.0-1b-speech-Q4_K_M.gguf"
        ),
        sha256="eacd9bc06063a3fdb6bf904a2a82c2646d504922c6e4e5eb1988d61e780884d1",
        size=1_602_904_800,
    ),
    PinnedModel(
        filename="granite-speech-4.1-2b-nar-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/"
            "c186f8d7ed6c67b947b90fd8ed9587be971766fa/granite-speech-4.1-2b-nar-Q4_K_M.gguf"
        ),
        sha256="68537a51d4d6d5a6be6e50f735a08af16646afba70fe8ab80dfe60ba7623325f",
        size=1_560_008_832,
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
        request = urllib.request.Request(model.url)  # noqa: S310 - pinned https URL
        # Gated repos (medasr) need a token of an account that accepted the
        # terms; it is sent to huggingface.co only.
        token = os.environ.get("HF_TOKEN", "")
        if token and model.url.startswith("https://huggingface.co/"):
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as out:  # noqa: S310
            shutil.copyfileobj(response, out, length=1 << 20)
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
