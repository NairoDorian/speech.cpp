#!/usr/bin/env python3
"""analyze_tts_compare.py — compare official CLI WAV vs CAPI raw-f32 output.

For each model in <outdir>, reads:
  <name>.official.wav   (16-bit PCM, written by official CLI)
  <name>.capi.wav.f32   (raw mono f32, dumped by capi_tts_test)
  <name>.official.log / <name>.capi.log (timing lines)

Prints a tab-separated comparison table: timing, playback-validity,
signal stats, and (when both outputs exist and are same rate) how close
the two waveforms are. Exit code 0.
"""
import argparse
import math
import os
import re
import struct
import wave

import numpy as np


def read_wav16(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype="<i2")
    if ch > 1:
        data = data[::ch]  # first channel only (mono downmix for analysis)
    return data.astype(np.float64) / 32768.0, sr


def read_f32(path):
    data = np.fromfile(path, dtype="<f4")
    return data.astype(np.float64)


def read_log(path):
    kv = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", errors="replace")
        for line in text.splitlines():
            m = re.match(r"^([a-z_]+)=(.+)$", line.strip())
            if m:
                kv[m.group(1)] = m.group(2)
    return kv


def stats(x):
    valid = x[np.isfinite(x)]
    n = len(x)
    nan = int(np.isnan(x).sum())
    inf = int(np.isinf(x).sum())
    clip = int((x > 1.0).sum() + (x < -1.0).sum())
    if len(valid) == 0:
        return dict(n=n, nan=nan, inf=inf, clip=clip, rms=float("nan"),
                    peak=float("nan"), dc=float("nan"), noise_db=float("nan"))
    rms = float(np.sqrt(np.mean(valid ** 2)))
    peak = float(np.max(np.abs(valid)))
    dc = float(np.mean(valid))
    # noise floor: RMS of the quietest 200ms window
    win = 0.2
    w = max(1, int(win * 0))  # placeholder
    # use first 100ms after a 100ms offset and last 100ms
    tail = int(0.1 * 0)
    # simpler: RMS of first and last 0.15s (assume lead/trail silence)
    edge = max(1, int(len(valid) * 0.02))
    front = valid[:edge]
    back = valid[-edge:]
    quiet = np.concatenate([front, back])
    noise = float(np.sqrt(np.mean(quiet ** 2))) if len(quiet) else float("nan")
    noise_db = 20.0 * math.log10(noise + 1e-12) if noise > 0 else float("-inf")
    return dict(n=n, nan=nan, inf=inf, clip=clip, rms=rms, peak=peak, dc=dc,
                noise_db=noise_db)


def compare_waveforms(a, b, sr):
    """Return (sample_rate, dur_a, dur_b, diff_db, corr). diff_db = RMS of
    (a-b) relative to RMS of a, in dB. corr = normalized cross-correlation."""
    n = min(len(a), len(b))
    if n < 1000:
        return None
    a, b = a[:n], b[:n]
    va, vb = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(va) < 1000 or len(vb) < 1000:
        return None
    rms_a = math.sqrt(np.mean(va ** 2))
    if rms_a < 1e-9:
        return None
    diff = np.sqrt(np.mean((a - b) ** 2)) / rms_a
    diff_db = 20.0 * math.log10(diff + 1e-12)
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) == len(b) else float("nan")
    return diff_db, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    outdir = args.outdir

    names = []
    for f in sorted(os.listdir(outdir)):
        if f.endswith(".official.wav"):
            names.append(f[: -len(".official.wav")])

    rows = []
    for name in names:
        owav = os.path.join(outdir, name + ".official.wav")
        cf32 = os.path.join(outdir, name + ".capi.wav.f32")
        olog = read_log(os.path.join(outdir, name + ".official.log"))
        clog = read_log(os.path.join(outdir, name + ".capi.log"))

        row = {"name": name}
        row["o_wall_ms"] = olog.get("metrics.wall_ms")
        row["o_rtf"] = olog.get("metrics.rtf")
        row["o_sr"] = olog.get("metrics.sample_rate")
        row["o_err"] = olog.get("audiocpp_cli failed")
        row["c_load_ms"] = clog.get("load_ms")
        row["c_synth_ms"] = clog.get("synth_ms")
        row["c_sr"] = clog.get("sample_rate")
        row["c_n"] = clog.get("n_samples")
        row["c_err"] = clog.get("error_message")

        if os.path.exists(owav):
            try:
                o_audio, o_sr = read_wav16(owav)
                os_ = stats(o_audio)
                row.update({f"o_{k}": v for k, v in os_.items()})
                row["o_dur_s"] = round(len(o_audio) / o_sr, 3)
                row["o_sr"] = o_sr
            except Exception as e:
                row["o_parse_err"] = str(e)
        if os.path.exists(cf32):
            c_audio = read_f32(cf32)
            cs = stats(c_audio)
            row.update({f"c_{k}": v for k, v in cs.items()})
            row["c_dur_s"] = round(len(c_audio) / (int(row["c_sr"]) if row["c_sr"] else 1), 3) \
                if row.get("c_sr") else None

        # waveform agreement when both exist and rates match
        if (os.path.exists(owav) and os.path.exists(cf32)
                and row.get("o_sr") and row.get("c_sr")
                and int(row["o_sr"]) == int(row["c_sr"])):
            cmp = compare_waveforms(o_audio, c_audio, int(row["o_sr"]))
            if cmp:
                row["diff_db"], row["corr"] = cmp
        rows.append(row)

    if args.json:
        import json
        print(json.dumps(rows, indent=2, default=str))
        return

    hdr = ("model | o_wall_ms | o_rtf | c_load_ms | c_synth_ms | "
           "o_sr | c_sr | o_dur | c_dur | o_rms | c_rms | o_peak | c_peak | "
           "o_nan | c_nan | o_clip | c_clip | o_noise_db | c_noise_db | "
           "diff_db | corr | o_err | c_err")
    print(hdr)
    for r in rows:
        print(" | ".join(str(r.get(k, "")) for k in [
            "name", "o_wall_ms", "o_rtf", "c_load_ms", "c_synth_ms",
            "o_sr", "c_sr", "o_dur_s", "c_dur_s", "o_rms", "c_rms",
            "o_peak", "c_peak", "o_nan", "c_nan", "o_clip", "c_clip",
            "o_noise_db", "c_noise_db", "diff_db", "corr", "o_err", "c_err"]))


if __name__ == "__main__":
    main()
