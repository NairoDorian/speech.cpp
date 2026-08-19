#!/usr/bin/env python3
"""Generate the pinned Silero VAD v6.2 safetensors in speech.cpp's native format.

Sole source: the official snakers4/silero-vad GitHub repo, tag v6.2, jit
checkpoint (src/silero_vad/data/silero_vad.jit). The jit's 16k submodule weights
are remapped to the engine runtime keys and written as safetensors.

The engine runtime (src/models/silero_vad/runtime.cpp) loads a safetensors with
tinygrad-style keys (stft_conv.weight, conv1-4, lstm_cell.*, final_conv.*).

IMPORTANT: the upstream tinygrad safetensors (snakers4/silero-vad master
src/silero_vad/data/silero_vad_16k.safetensors, added 2025-12-10) is a divergent
export that does NOT match the v6.2 jit weights (see snakers4/silero-vad
issue #768) - do NOT use it as the engine asset.

Verified 2026-08-19:
  safetensors sha D7FB67A5B4DE0414A0178270EE81439B9E3067C883C304AA758C247D50FFD79D
                  (generated from official v6.2 jit, 15/15 tensors, max_diff 0)

Usage:  uv run --with safetensors --with numpy --with torch --with torchvision scripts/fetch_silero_vad.py
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import safetensors.numpy
import torch

PINNED_ST_SHA256 = "D7FB67A5B4DE0414A0178270EE81439B9E3067C883C304AA758C247D50FFD79D"

V62_JIT_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2/src/silero_vad/data/silero_vad.jit"

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets/framework/models/silero_vad"
ST_OUT = ASSET_DIR / "silero_vad_16k.safetensors"

# audio.cpp runtime key -> v6.2 jit submodule key (16k branch)
JIT_MAP = {
    "stft_conv.weight": "_model.stft.forward_basis_buffer",
    "conv1.weight": "_model.encoder.0.reparam_conv.weight",
    "conv1.bias": "_model.encoder.0.reparam_conv.bias",
    "conv2.weight": "_model.encoder.1.reparam_conv.weight",
    "conv2.bias": "_model.encoder.1.reparam_conv.bias",
    "conv3.weight": "_model.encoder.2.reparam_conv.weight",
    "conv3.bias": "_model.encoder.2.reparam_conv.bias",
    "conv4.weight": "_model.encoder.3.reparam_conv.weight",
    "conv4.bias": "_model.encoder.3.reparam_conv.bias",
    "lstm_cell.weight_ih": "_model.decoder.rnn.weight_ih",
    "lstm_cell.weight_hh": "_model.decoder.rnn.weight_hh",
    "lstm_cell.bias_ih": "_model.decoder.rnn.bias_ih",
    "lstm_cell.bias_hh": "_model.decoder.rnn.bias_hh",
    "final_conv.weight": "_model.decoder.decoder.2.weight",
    "final_conv.bias": "_model.decoder.decoder.2.bias",
}

EXPECTED_SHAPES = {
    "stft_conv.weight": (258, 1, 256),
    "conv1.weight": (128, 129, 3),
    "conv1.bias": (128,),
    "conv2.weight": (64, 128, 3),
    "conv2.bias": (64,),
    "conv3.weight": (64, 64, 3),
    "conv3.bias": (64,),
    "conv4.weight": (128, 64, 3),
    "conv4.bias": (128,),
    "lstm_cell.weight_ih": (512, 128),
    "lstm_cell.weight_hh": (512, 128),
    "lstm_cell.bias_ih": (512,),
    "lstm_cell.bias_hh": (512,),
    "final_conv.weight": (1, 128, 1),
    "final_conv.bias": (1,),
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def regenerate_safetensors(force: bool) -> bool:
    """Generate the engine-native v6.2 safetensors from the official v6.2 jit."""
    if ST_OUT.exists() and not force:
        if sha256_of(ST_OUT) == PINNED_ST_SHA256:
            print(f"already up to date: {ST_OUT}")
            return True
        print(f"hash mismatch on existing safetensors, regenerating: {ST_OUT}")

    with tempfile.TemporaryDirectory() as td:
        jit_path = Path(td) / "silero_vad_v62.jit"
        try:
            with urllib.request.urlopen(V62_JIT_URL, timeout=120) as resp:
                with jit_path.open("wb") as f:
                    shutil.copyfileobj(resp, f)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: failed to download v6.2 jit from GitHub: {exc}", file=sys.stderr)
            return False
        try:
            jit = torch.jit.load(str(jit_path), map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: failed to load jit: {exc}", file=sys.stderr)
            return False
        sd = jit.state_dict()
        tensors = {}
        for out_key, jit_key in JIT_MAP.items():
            if jit_key not in sd:
                print(f"ERROR: missing jit key {jit_key}", file=sys.stderr)
                return False
            arr = sd[jit_key].detach().cpu().numpy().astype("float32")
            if arr.shape != EXPECTED_SHAPES[out_key]:
                print(f"ERROR: shape {arr.shape} for {out_key}, expected {EXPECTED_SHAPES[out_key]}", file=sys.stderr)
                return False
            tensors[out_key] = arr
        tmp = ST_OUT.with_suffix(".tmp")
        safetensors.numpy.save_file(tensors, str(tmp))
        actual = sha256_of(tmp)
        if actual != PINNED_ST_SHA256:
            print(f"ERROR: regenerated sha256 {actual} != pinned {PINNED_ST_SHA256}", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(ST_OUT)
        print(f"generated {ST_OUT} ({ST_OUT.stat().st_size} bytes)")
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate even if the hash matches")
    args = parser.parse_args()

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    return 0 if regenerate_safetensors(args.force) else 1


if __name__ == "__main__":
    sys.exit(main())