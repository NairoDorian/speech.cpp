# Benchmarking and port acceptance

> **Binding** (decision V7-D9, user directive 2026-09-26; `PLAN.md` rules R5 and R6).
>
> **The rule: no model enters or changes in speech.cpp if it is slower or less accurate than the parents that have it.**
>
> Terms and metric definitions (WER/CER and normalization, RTF vs RTFx, TTFA, DER, SIM, …) and the standard datasets per task are in [`glossary.md`](glossary.md).
>
> The tooling is the author's own transcribe.cpp harness (`scripts/bench/`, `scripts/wer/`), ported with extra arms for audio.cpp and speech.cpp (`PLAN.md` C0.6).

---

## 1. The acceptance rule

Every port, and every change that touches a family's numerics or kernels, is compared against **each parent that has the family**:
- audio.cpp: its `audiocpp_cli` or engine on the same GGUF;
- transcribe.cpp: `transcribe-bench` / `transcribe-cli` on the same GGUF.

All arms run on the **same machine, backend, thread count and GGUF bytes** (record the sha256).

| Metric | Pass condition |
|---|---|
| **WER / CER** (benchmark subset, per language and macro-average) | **≤ the best parent.** The same GGUF with the same decode should give identical transcripts. Any increase is a failure unless its cause is recorded in `docs/LEDGER.md` and accepted by the user. |
| **Speed**: RTF mean of runs 2..N (for TTS: RTF and time-to-first-audio) | **≤ the best parent**, within the noise floor. The noise floor is the larger of the two arms' run-to-run spread and 2 %. Slower beyond the floor is a failure (V7-D8: the author's transcribe.cpp optimizations must survive). |
| **Peak memory** (RSS / VRAM) | ≤ 1.1 × the best parent. |
| **Cold start** (run 1: load + first inference) | Reported and analyzed, never averaged. A regression here is a finding to explain, because apps feel it. |

**If only one parent has the family**, compare with that one. **If neither does** (CrispASR-derived or new), compare WER with the PyTorch reference on the same subset (C-R2), and speed with CrispASR as a second opinion.

## 2. Run protocol

1. **At least 3 runs per cell. Run 1 is the warm-up:** exclude it from every average, but record it (load time, first-inference latency, first-run RTF) and look at it.
2. **Report the mean of runs 2..N**, with min/max:
   - N = 3 in the development loop;
   - N = 5 for acceptance decisions;
   - **never more than 10**.
3. **Interleave arms** (A B A B …, never A A A B B B). Use an idle machine: laptop plugged in, nothing else on the GPU, nearly-full disks avoided (LESSONS F5).
4. **Same thread count on every arm**, stated explicitly. Output can depend on threads (whisper-tiny did: checkpoint §2.2).
5. **Inside the noise floor means equal** (L12). Never claim a win or a loss inside it.

## 3. Datasets: short by design

| Task | Development subset (N = 3 runs) | Acceptance subset (N = 5 runs) |
|---|---|---|
| ASR, multilingual | **FLEURS test, first 10 utterances × `core6`**: en, fr, de, es, ru, zh (CER), restricted to the languages the model supports | first 20 utterances × `core6`, plus any language the model is marketed for (e.g. ja, ar, it, pt) |
| ASR, English-only | FLEURS en × 10 + the in-tree LibriSpeech 4-clip corpus | FLEURS en × 20 + LibriSpeech test-clean × 50 (fixed subset) |
| Streaming ASR | as above, streamed in 160–320 ms feeds | + time-to-first-partial, end-of-speech → final, streamed-vs-offline divergence |
| TTS | 6 fixed sentences per supported language (of `core6`) | 12 sentences; round-trip WER with one fixed ASR model + RTF + time-to-first-audio |

- Subsets are **deterministic** (transcribe.cpp `scripts/wer/subset.py`: first-N in manifest order) and fetched by `scripts/wer/ingest.py fleurs --lang <x>`. Audio stays out of git (`models/`-style ignore). FLEURS is CC-BY-4.0.
- The in-tree quick clip (`assets/asr_validation/quick/`) is a **smoke test, not a benchmark**. It proves the model runs (the ctest quick tier); it never proves a port is good.
- **Keep it small.** One shipped quant per family (the one FreeSpeech bundles). Add F16 only to diagnose. Never every quant × every language × 10 runs.

## 4. Backend matrix: CPU during development, all four before acceptance

| When | Backends | Why |
|---|---|---|
| Edit / build / debug loop | **CPU only** (`build-cpu-core`, `build-cpu-asr-abi`) | Fastest to build and verify (AGENTS.md). |
| Before a port is accepted, an arch retires, or a family joins Tier A; also after a kernel/ggml change | **CPU, CUDA (NVIDIA), Vulkan on NVIDIA, Vulkan on Intel (iGPU)** | Both FreeSpeech targets run on varied hardware, and the Vulkan paths differ per vendor. |
| When phase E6 lands | + **Android arm64** (device or emulator) | FreeSpeech Android (#2). |

**On this machine:** i9-13900H CPU, RTX 4070 Laptop (CUDA sm_89, Vulkan) and Intel Iris Xe (Vulkan).
- Vulkan device selection: `GGML_VK_VISIBLE_DEVICES=<index>`. Record the index → device mapping in the report.
- Trees: `build-cpu-*` and `build-cuda-core` exist. **`build-vulkan` must be added** (`GGML_VULKAN=ON`; `PLAN.md` C0.6).
- Parent reference builds live **outside the repos** (`../_ref_builds/<parent>-<backend>/`), so the sibling checkouts stay untouched.

## 5. Report format

One markdown file per family: `docs/reports/bench/<family>.md`. Raw JSON goes under `reports/bench/`, which is gitignored.

For each (variant, quant, backend) row, report:
- WER/CER per language and macro-average;
- RTF mean(2..N) with min–max;
- run-1 RTF and load time (ms);
- peak RAM/VRAM;
- **the same columns for audio.cpp and transcribe.cpp**;
- a verdict (`PASS` / `FAIL: <metric>`).

Also state the machine, driver versions, thread count, N, GGUF sha256 and date. These rows feed the future scoreboard (PLAN SC6).

## 6. Budget: keep development fast

| Loop | What runs | Target time |
|---|---|---|
| Edit loop | `ctest -L quick` on CPU | ≈ 4 min for the whole ASR quick tier |
| Port candidate | 1 family × 1 quant × development subset × CPU × N = 3, against both parents | ≈ 10–20 min |
| Acceptance | 1 family × 1 quant × acceptance subset × 4 backends × N = 5 | ≈ 1–2 h, run rarely |
