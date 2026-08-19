#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "scipy>=1.14",
#     "safetensors>=0.4",
#     "soundfile>=3.0",
#     "torch>=2.3",
# ]
# ///
"""
dump_reference_silero_vad.py - reference dump for the Silero VAD v6.2 model.

Reads the engine-native safetensors (scripts/fetch_silero_vad.py output at
assets/framework/models/silero_vad/silero_vad_16k.safetensors) and re-implements
the forward pass bit-for-bit against the architecture in
src/models/silero_vad/runtime.cpp, dumping intermediate tensors in the validate.py
dump contract:

    <name>.f32   raw little-endian fp32, row-major (C order)
    <name>.json  shape, dtype, layout

Dump point names mirror the C++ graph tensor names (prefixed vad.) so the
C++ debug dumper can be wired to the same keys once Phase 4 lands dump calls
in src/models/silero_vad/runtime.cpp.

Usage:
    uv run --project scripts/envs/silero_vad \\
        scripts/dump_reference_silero_vad.py \\
        --model assets/framework/models/silero_vad/silero_vad_16k.safetensors \\
        --audio samples/jfk.wav \\
        --out build/validate/silero_vad/silero_vad_v6.2/jfk/ref \\
        --torch-threads 1

Stage subcommand is ignored (silero_vad has no encoder/decode split) —
accepted for validate.py convention-compatibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import safetensors.numpy
import soundfile as sf
import torch
import torch.nn.functional as F


# Architecture constants — mirror src/models/silero_vad/runtime.cpp.
K_SAMPLE_RATE = 16000
K_CHUNK_SAMPLES = 512
K_CONTEXT_SAMPLES = 64
K_INPUT_SAMPLES = K_CHUNK_SAMPLES + K_CONTEXT_SAMPLES  # 576
K_CUTOFF = 129   # 129 freq bins (n_fft=256 → n_freq=129)
K_HIDDEN_SIZE = 128


def dump_tensor(out_dir: Path, name: str, array: np.ndarray) -> None:
    """Write (<name>.f32, <name>.json) in the validate.py dump contract."""
    flat = np.ascontiguousarray(array, dtype=np.float32).ravel()
    shape = list(array.shape) if array.ndim > 0 else [int(flat.size)]
    payload = {
        "name": name,
        "dtype": "f32",
        "layout": "row-major",
        "shape": shape,
        "source": "silero_vad_reference",
    }
    (out_dir / f"{name}.f32").write_bytes(flat.tobytes())
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def load_weights(path: Path) -> dict[str, np.ndarray]:
    """Load the engine-native safetensors produced by fetch_silero_vad.py."""
    arrays = safetensors.numpy.load_file(str(path))
    return {k: v.astype(np.float32) for k, v in arrays.items()}


def lstm_cell(
    x: torch.Tensor,
    h: torch.Tensor,
    c: torch.Tensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor,
    b_hh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-step LSTM cell matching PyTorch's LSTMCell semantics.

    The C++ LSTMCellModule in src/models/silero_vad/runtime.cpp implements the
    standard C = f*s + i*g gate layout; this mirrors it exactly.
    """
    gates = F.linear(x, w_ih, b_ih) + F.linear(h, w_hh, b_hh)
    i, f, g, o = gates.chunk(4, dim=-1)
    c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    h_new = torch.sigmoid(o) * torch.tanh(c_new)
    return h_new, c_new


def dump_for_chunk(
    out_dir: Path,
    weights: dict[str, np.ndarray],
    chunk_input: np.ndarray,  # shape (576,) float32
    h_state: np.ndarray,      # shape (128,) float32
    c_state: np.ndarray,      # shape (128,) float32
) -> tuple[np.ndarray, np.ndarray]:
    """Run one chunk through the reference forward pass and dump intermediates.

    Returns the updated (hidden, cell) states for the next chunk.
    """
    w = weights

    # --- STFT (reflect-pad + conv with forward basis) ---
    # ReflectPad1d({0, kContextSamples}) pads 64 samples at the end.
    padded = np.pad(chunk_input, (0, K_CONTEXT_SAMPLES), mode="reflect")
    padded_t = torch.from_numpy(np.ascontiguousarray(padded, dtype=np.float32)).unsqueeze(0)  # (1, 640)

    stft_basis = torch.from_numpy(w["stft_conv.weight"])  # (258, 1, 256)
    stft = F.conv1d(padded_t.unsqueeze(1), stft_basis, stride=128)  # (1, 258, 4)
    stft_real = stft[:, :K_CUTOFF, :]  # (1, 129, 4)
    stft_imag = stft[:, K_CUTOFF:, :]  # (1, 129, 4)
    stft_mag = torch.sqrt(stft_real ** 2 + stft_imag ** 2)  # (1, 129, 4)

    dump_tensor(out_dir, "vad.stft_magnitude", stft_mag.squeeze(0).numpy())

    # --- Encoder conv stack ---
    x = stft_mag
    for i, (conv_w_key, conv_b_key, dump_name) in enumerate([
        ("conv1.weight", "conv1.bias", "vad.conv1.out"),
        ("conv2.weight", "conv2.bias", "vad.conv2.out"),
        ("conv3.weight", "conv3.bias", "vad.conv3.out"),
        ("conv4.weight", "conv4.bias", "vad.conv4.out"),
    ]):
        stride = [1, 2, 2, 1][i]
        conv_w = torch.from_numpy(w[conv_w_key])  # (out_c, in_c, 3)
        conv_b = torch.from_numpy(w[conv_b_key])  # (out_c,)
        x = F.conv1d(x, conv_w, conv_b, stride=stride, padding=1)
        x = F.relu(x)
        dump_tensor(out_dir, dump_name, x.squeeze(0).numpy())

    # conv4 is reshaped to (1, 128) — the LSTM input.
    conv4_flat = x.reshape(1, K_HIDDEN_SIZE)

    # --- LSTM cell ---
    h_t = torch.from_numpy(h_state).unsqueeze(0)  # (1, 128)
    c_t = torch.from_numpy(c_state).unsqueeze(0)  # (1, 128)
    w_ih = torch.from_numpy(w["lstm_cell.weight_ih"])  # (512, 128)
    w_hh = torch.from_numpy(w["lstm_cell.weight_hh"])  # (512, 128)
    b_ih = torch.from_numpy(w["lstm_cell.bias_ih"])    # (512,)
    b_hh = torch.from_numpy(w["lstm_cell.bias_hh"])    # (512,)

    h_out, c_out = lstm_cell(conv4_flat, h_t, c_t, w_ih, w_hh, b_ih, b_hh)
    dump_tensor(out_dir, "vad.hidden_out", h_out.squeeze(0).numpy())

    # --- Final conv + sigmoid ---
    hidden_relu = F.relu(h_out).unsqueeze(-1)  # (1, 128, 1)
    final_w = torch.from_numpy(w["final_conv.weight"])  # (1, 128, 1)
    final_b = torch.from_numpy(w["final_conv.bias"])    # (1,)
    prob = torch.sigmoid(F.conv1d(hidden_relu, final_w, final_b))  # (1, 1, 1)
    prob = prob.reshape(1)  # (1,)

    dump_tensor(out_dir, "vad.probs", prob.squeeze().numpy())

    return h_out.squeeze(0).numpy(), c_out.squeeze(0).numpy()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", nargs="?", default="encoder",
                   help="Stage subcommand (accepted for validate.py convention; "
                        "silero_vad has no split — ignored)")
    p.add_argument("--model", required=True, help="Path to silero_vad_16k.safetensors")
    p.add_argument("--audio", required=True, help="Path to input WAV (16 kHz mono)")
    p.add_argument("--out", required=True, help="Output directory for dump files")
    p.add_argument("--torch-threads", type=str, default="1",
                   help="Intra-op threads for torch (default: 1 for determinism)")
    args = p.parse_args()

    # Validate stage — silero_vad ignores the encoder/decode split.
    if args.stage not in ("encoder", "decode"):
        print(f"error: stage must be 'encoder' or 'decode', got '{args.stage}'", file=sys.stderr)
        return 2

    torch.set_num_threads(int(args.torch_threads))

    weights = load_weights(Path(args.model))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load audio — must be 16 kHz mono.
    audio, sr = sf.read(str(args.audio), dtype="float32")
    if sr != K_SAMPLE_RATE:
        print(f"error: audio sample rate {sr} != required {K_SAMPLE_RATE}", file=sys.stderr)
        return 2
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono

    # Process in 512-sample chunks with 64-sample reflect context.
    # Context is seeded from the preceding chunk's tail (matching the C++
    # offline path which pre-fills the first 64-sample context by reflecting
    # the start of the signal).
    h_state = np.zeros(K_HIDDEN_SIZE, dtype=np.float32)
    c_state = np.zeros(K_HIDDEN_SIZE, dtype=np.float32)

    # Pre-fill 64-sample context by reflecting the first 64 samples.
    context = audio[:K_CONTEXT_SAMPLES] if len(audio) < K_CONTEXT_SAMPLES else audio[:K_CONTEXT_SAMPLES]
    if len(context) < K_CONTEXT_SAMPLES:
        context = np.pad(context, (0, K_CONTEXT_SAMPLES - len(context)), mode="reflect")

    offset = 0
    chunk_idx = 0
    while offset + K_CHUNK_SAMPLES <= len(audio):
        chunk = np.zeros(K_INPUT_SAMPLES, dtype=np.float32)
        chunk[:K_CONTEXT_SAMPLES] = context
        chunk[K_CONTEXT_SAMPLES:] = audio[offset:offset + K_CHUNK_SAMPLES]

        # Dump each chunk into a per-chunk subdirectory.
        chunk_dir = out_dir / f"chunk_{chunk_idx:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        h_state, c_state = dump_for_chunk(chunk_dir, weights, chunk, h_state, c_state)

        # Shift context: last 64 samples of this chunk become the next context.
        context = audio[offset + K_CHUNK_SAMPLES - K_CONTEXT_SAMPLES:offset + K_CHUNK_SAMPLES]
        offset += K_CHUNK_SAMPLES
        chunk_idx += 1

    if chunk_idx == 0:
        print("error: audio shorter than one chunk; nothing to dump", file=sys.stderr)
        return 2

    print(f"reference dumps written to {out_dir} ({chunk_idx} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
