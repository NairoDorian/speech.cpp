# Progress — Unified_Audio.cpp (speech.cpp ggml fork) merge & improve

Status snapshot: **speech.cpp transcribes speech, end to end, through the
public C ABI — and the new gate's first companion investigation found a fourth
dropped ggml fork delta (unmarked, like 0004), now restored as patch 0006 and
re-certified on CPU and CUDA. Both suites moved one test greener; the
sync-ggml invariant is re-verified with six patches in the stack.**
Date: 2026-08-20 (second work session this date)

## Repo layout (important, non-obvious)
`Unified_Audio.cpp/` is a **plain container directory with no git repo of its
own**. It holds exactly three things, each an independent repository:

| Folder | Role |
|---|---|
| `speech.cpp/` | the active development repo (the ggml/audio.cpp fork). **All merge work, and this log, live here.** Remote: `NairoDorian/speech.cpp`, upstream `0xShug0/audio.cpp`. |
| `audio.cpp/` | upstream reference — read from, not developed in. This session it also served as the **discriminating experiment host**: same test + same fixtures built against the fork ggml on the same machine/compiler. |
| `transcribe.cpp/` | merge source — read from, not developed in |

Build trees are scratch dirs under `C:/Users/Z/AppData/Local/Temp/opencode/`:
`sp_bridge` (CPU, full model set, unified ABI + arches, tests), `sp_cuda`
(CUDA, core set), `audiocpp_flashsr` (audio.cpp reference build, new this
session). `CMakePresets.json` now reproduces the first two as `cpu-full` /
`cuda` (plus a `cpu-core` lean preset) for anyone without those dirs.

## Overall progress (toward "Unified_Audio transcribes on CPU")
| Area | Status | % |
|---|---|---|
| Merge: ggerganov/ggml → bundled `external/ggml` (pin 8c63e709) + fork patches | Done | 100% |
| Reproducibility (`sync-ggml.sh` + patches **0001–0006** → empty diff) | **Re-verified with 0006; script hardened vs CRLF patches** | 100% |
| Phase-2 CPU compute + CUDA parity for the three fork-only ops | Done (prior sessions) | 100% |
| Fork-delta audit | **Corrected: marker grep is necessary, NOT sufficient — 0004/0006 were unmarked. Numeric golden gates are the real audit.** | ongoing by construction |
| ABI offline + streaming paths | Verified, real CTest gates | 100% |
| **End-to-end ASR transcription validation (real model, real WER)** | **Done — 1.45% corpus WER through the C ABI** | **100%** |
| `flashsr_utility_test` (was failing since the convergence) | **Root-caused + fixed (patch 0006), green on CPU and CUDA** | 100% |
| **Project-wide (functional CPU transcribe)** | | **~85%** |

## DONE this session

### 1. End-to-end ASR validation — the top blocker, closed
`tests/asr_e2e_wer_test.cpp` + CTest gate `asr_e2e_wer_test`: loads a real
GGUF through `transcribe_open()`, transcribes the four in-tree LibriSpeech
fixtures, scores **corpus WER** against the references with LibriSpeech-style
normalization, and gates at 10%. Links ONLY `transcribe.dll`, like a language
binding. Model: **moonshine-tiny Q8_0** (34 MB, MIT, arch already in
`src/runtime/arch/moonshine`, transcribe.cpp-validated at 4.60% on the full
test-clean split), sha256-pinned by the new stdlib-only
`scripts/fetch_asr_test_model.py` into gitignored `models/`; the test skips
(exit 2) while the file is absent, so the gate arms itself on fetch with no
reconfigure.

Measured (CPU, i9-13900H, clang-cl Release): **corpus WER 1.45%** — 1 edit in
69 words (`FORWARDED`→`VOTED`, consistent with the model's published 4.6%),
RTF 0.033. Report: `docs/reports/asr_e2e_wer_gate.md`. The WAV reader moved
to header-only `tests/abi_test_wav.h`, shared with `abi_stream_hello`.

Offline moonshine reports `supports_streaming == false`, so `abi_stream_hello`
correctly skips it (streaming stays validated via silero_vad segments); the
gap that remains is streaming *text* (see NEXT).

### 2. `flashsr_utility_test` — not a tolerance problem: the 4th dropped fork delta
progress' previous NEXT #3 said "worth a look on its own terms". The look:

- The failure (1.7e-3 max vs a 2e-4 bound) is **identical on CPU and CUDA** —
  the signature of a backend-independent graph-builder delta, not a kernel bug.
- The fixtures are onnxruntime-generated (implementation-independent truth)
  and byte-identical in both trees; the same test **built from audio.cpp's
  fork ggml passes on this machine and compiler**. That was the discriminating
  experiment (`audiocpp_flashsr` build).
- Cause: upstream 0.20.2's `ggml_conv_1d` / `ggml_conv_1d_dw` / `ggml_conv_2d`
  force im2col activations to **F16** unless the weight is BF16; the fork
  keeps them in the **weight's type**. Every framework conv weight is F32, so
  the convergence silently demoted every framework convolution to F16
  activations.
- Restored as `patches/ggml/0006-conv-im2col-in-weight-type.patch`, tagged
  `MINITTS_CONV_IM2COL_WEIGHT_TYPE` (the fork left it UNMARKED — see §4).
  `ggml_conv_2d_dw` deliberately untouched: the fork kept upstream's F16 there.
- After the restore: `flashsr_utility_test` green on **both** backends for the
  first time since the convergence; the WER gate unchanged at 1.45%; no other
  test moved on either suite.

### 3. Reproducibility re-verified with 0006 — after a self-inflicted scare
The first `sync-ggml.sh` run died applying 0006. The patch was correct; the
freshly-written *file* carried CRs in its hunk lines, which cannot match the
LF stage (`git archive` honors `eol=lf`). Re-materializing the file from the
index fixed it, and `scripts/sync-ggml.sh` now normalizes every patch to LF
before applying so the trap is closed for the next author. Final state:
sync + 0001–0006 → `git diff --ignore-cr-at-eol -- external/ggml` **empty**
(the plain-diff "13 paths" remain the documented CRLF-only Windows artifact).

### 4. The audit doctrine is corrected (plan R11)
R10 claimed the `MINITTS_*` grep "enumerates the class" of behavioural fork
deltas. **It does not**: 0004 (two-sided broadcast) and 0006 (conv im2col
type) both carried no marker. What actually catches the unmarked class:
numeric golden gates whose references are implementation-independent
(onnxruntime fixtures caught 0006; the WER gate now guards ASR the same way),
and a hunk-level diff of the fork tree against its own upstream base. Recorded
in `external/ggml/UPSTREAM` and plan R11.

### 5. Quality-of-life
- `CMakePresets.json`: `cpu-full` / `cpu-core` / `cuda` configure+build+test
  presets matching the validated configurations (closes a LEFT-TO-DO item).
- Fetch-script convention: `uv run scripts/fetch_asr_test_model.py` (uv is
  the Python runner on this machine; the script is stdlib-only so any
  Python 3 works).

## Test suite state
Both backends moved one test greener than the documented baseline; nothing
else changed.

**CPU build** (`sp_bridge`: full set, unified ABI + arches) — **57 tests**
(56 + the new WER gate), 5 failures, all pre-existing environment/asset
issues: `model_spec_system_test`, `fun_asr_nano_assets_test`,
`scaled_dot_product_attention_test` (needs CUDA, absent here),
`asr_standalone_gguf_test` (needs citrinet+hviske assets),
`server_model_installer_test`. Newly green: **`asr_e2e_wer_test`**,
**`flashsr_utility_test`**.

**CUDA build** (`sp_cuda`: core set, sm_89) — 46 tests, **3 failures** (was
4): the same `model_spec_system_test`, `fun_asr_nano_assets_test`,
`server_model_installer_test`. Newly green: **`flashsr_utility_test`**.

## NEXT (highest value first)
1. **Streaming ASR text validation.** The streaming surface is exercised only
   against silero_vad segments. `src/runtime/arch/moonshine_streaming` exists;
   find/pin a small streaming-family GGUF (moonshine-streaming-tiny would be
   ideal if published) and extend the gate — or run `abi_stream_hello`'s
   model through a streaming arch and compare streamed text to offline text.
2. **`fun_asr_nano_assets_test` / `model_spec_system_test` /
   `server_model_installer_test`** — the three failures shared by both
   backends. Filed as "environment/asset issues" for three sessions now;
   after this session's flashsr lesson ("missing assets" turned out to be a
   real numeric regression), each deserves one honest look.
3. **`asr_standalone_gguf_test`** — needs citrinet_asr + hviske_asr GGUFs;
   both families have model_specs. Same download-and-pin pattern as the WER
   gate would make it a real gate too.
4. **Appendix I build matrix presets** — `cpu-core` preset exists now; wire
   the lean-build row into whatever CI lands (plan R10.5 wants "lean set +
   tests ON" first-class).

## LEFT TO DO (small)
- [ ] Formalize root `.gitmodules` so `git submodule update` works for the 3 embedded repos.
- [ ] `LNK4217` warnings: arch static libs see `TRANSCRIBE_API` as `dllimport`
      for same-DLL symbols. Benign but noisy.
- [ ] `external/ggml` mixed line endings (1145 CRLF blobs vs `eol=lf`):
      unchanged posture — leave alone; sync matches committed state on
      Windows; a Linux run would produce LF + a large diff.
- [x] ~~CMakePresets.json~~ (this session)

## Notes / decisions made this session
- A failure that is *identical* on CPU and CUDA is prior evidence for a
  backend-independent (graph-builder or weights-path) cause — that heuristic
  found patch 0006 and is cheap to apply to any future numeric mystery.
- "Missing assets" is not a diagnosis. Two of this repo's long-standing
  "asset" failures have now turned out to be something else entirely
  (flashsr: a real regression; the WER gap: an unfetched-by-design model).
- Golden fixtures are only as good as their reference's independence:
  onnxruntime fixtures caught what per-op CPU↔CUDA parity could not, because
  both backends inherited the same builder delta.
- The model-pinning pattern (sha256-pinned fetch script + gitignored
  `models/` + skip-not-fail test registration) is now established twice
  (silero_vad, moonshine); reuse it for citrinet/hviske (NEXT #3).
