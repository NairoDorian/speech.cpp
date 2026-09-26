# speech.cpp — agent conventions

**Read these first, in order:**
0. [`ARCHITECTURE.md`](ARCHITECTURE.md): the **generated map** of the whole tree (directories, sizes, families, entry points). Start here to find anything fast. For file contents, make a *scoped* pack: `bun scripts/arch/repomix.ts <paths…>`.
1. [`PLAN.md`](PLAN.md) → **North star**: the end goal and the rules R1–R11. Re-read it at the start of every session.
   - If a term or metric is unfamiliar (CTC, TDT, RTF vs RTFx, WER normalization, DER, SIM, …), see [`docs/glossary.md`](docs/glossary.md).
2. `bash scripts/status.sh` — the measured state.
3. [`PLAN.md`](PLAN.md) §0 and §2.3 — what to do next, and what is known-bad.
4. [`LESSONS.md`](LESSONS.md) — the headings. Every one of them is a mistake that already happened here.

Records live in [`docs/LEDGER.md`](docs/LEDGER.md): decisions, family state and deletions. Contribution policy and style are in `CONTRIBUTING.md`. **There are no other plan documents.** The pre-2026-09-26 plans are in `docs/archive/` and are history, not instructions.

## What speech.cpp is (V7-D7, user directives)

**One codebase, two build profiles:**

| Profile | Contains | Rule |
|---|---|---|
| **DEV (full)** | Every family, plus the server + WebUI, CLI, GGUF tools, model manager, tests, benches, verdict/parity harnesses | **Maximum functionality: every way to try and test every model.** Keep it working and extend it; new families should be testable here. |
| **BUNDLE (embed)** | `libspeech` + `speech.h` + Rust crate, with **only the selected families** (a composite, or a custom list down to one STT or TTS family) | What apps ship. Stripped of everything else: no server, WebUI, CLI, tools, tests or demo assets. |

- **DEV tools depend on the library, never the reverse.** Reject any change that makes `libspeech` need a DEV component, or a BUNDLE build that pulls one in.
- **The library itself** does no audio-device I/O (the host owns mic and speaker), and needs no Python, no network and no CWD-relative assets at runtime. It catalogues and verifies models; the app downloads them.
- **Primary consumers:**
  - **FreeSpeech desktop** (primary target #1): the successor of ZER0 (`../../Handy_V2`, Multi-STT: up to 9 concurrent STT sessions) and S2B2S (`../../STT_BRAIN_TTS/S2B2S`).
  - **FreeSpeech Android** (primary target #2, `../../Android_FreeSpeech`): Rust cdylib + JNI, arm64, small `.so`, low RSS, streaming partials, R2T2.
  - speech.cpp is meant for **both**. A change that works on only one of them is not done.
  - Both link transcribe.cpp today, so `transcribe.h` / the `transcribe-cpp` crate must keep working through the shim until they migrate.
- **Out of scope:** the LLM brain. It is llama.cpp on desktop and LiteRT-LM (Gemma 4) on Android; expose hooks, never embed it.
- **Before adding anything, ask:** does the BUNDLE profile need it, or only DEV? Put it on the right side of that line.

## Working protocol

- **Active repo:** `speech.cpp` only. The sibling checkouts are references: `../audio.cpp`, `../transcribe.cpp`, `../transcribe_cunba`, `../CrispASR`. Never edit them.
- **One phase at a time**, in PLAN order.
  - Do the first unchecked item of the current phase.
  - Do not start a later phase's work "while you're there". This is exactly how the plan got inverted in 2026-09 (LESSONS A7).
- **Tick a box only by running its gate** (L14).
  - `[~]` means the code is written but the gate has not been run. Never write `[x]` for that.
  - Counts come from `scripts/status.sh`, never from memory or old docs.
- **Architecture gates** (PLAN §5) run before and after every port (L15).
- **Phase exit:**
  1. Run the suites (below).
  2. Update PLAN §2.2 from `status.sh`.
  3. Tick the gated boxes.
  4. Add one dated `CHANGELOG.md` entry.
  5. Ask the user before starting the next phase.
- **Handover:** if you stop with uncommitted or unverified work, put that fact at the top of your CHANGELOG entry and of your final message (L17).
- **One writer per tree.** Do not run two agents that edit or move files in the same checkout at the same time.

## Pre-commit habit (V7-D10)

- **One-time setup per clone:** `git config core.hooksPath .githooks`, then `cd scripts/arch && bun install --frozen-lockfile`.
- On every commit the hook:
  - regenerates `ARCHITECTURE.md` with repomix and stages it;
  - rejects staged files with control bytes, absolute user paths in code, or `external/ggml` edits without a sync or patch.
  - It takes about 1 s.
- **Never hand-edit `ARCHITECTURE.md`.** To change a description, edit `DIR_DESCRIPTIONS` in `scripts/arch/gen-architecture.ts`. A new top-level area must get a description; the generator warns until it does.
- The hook never builds or tests. Builds and tests are the phase-exit routine below.

## Git

- Work on `main`. No feature branches.
- **Commit or push only when the user explicitly says so.**
  - When a suite goes green, *ask* for the go-ahead. Then commit on `main` and push `origin main`. Never force-push.
  - Commit in coherent units: one family per commit, with its CMake hunk, sources, spec and tests together. Never commit a `CMakeLists.txt` hunk without the files it references.
- Don't reformat unrelated code in a behaviour fix.
- Delete a branch only if it is merged (`git branch -d`). Surface unmerged ones to the user.

## Three sources — equal authority, different mechanics

- **audio.cpp** (`upstream` = `0xShug0/audio.cpp`, git parent) and **transcribe.cpp** (`../transcribe.cpp`, no git remote here) are **equal parents** (L13).
  - **transcribe.cpp here is the author's own fork** (`NairoDorian/transcribe.cpp`), where most of the author's engineering time went before speech.cpp. Its optimizations are **must-adopt**, and a port that loses one (slower RTF, more memory) is a regression (V7-D8).
  - audio.cpp is pure upstream: none of the author's work. Track it fully for its latest models and optimizations.
  - An improvement or dependency bump in either one is ours.
  - Both get the same audit by content and a disposition ledger.
  - The missing transcribe.cpp merge-base is a tooling limitation, not a hierarchy.
  - When the parents disagree, record the decision in `docs/LEDGER.md`. "audio.cpp is the fork base" never breaks the tie.
- **CrispASR** (`../CrispASR`) is reference parent #3 (V7-D2).
  - It is **mined by family, never merged.** Its code is read as a specification; the oracle stays PyTorch.
  - Triage by family in `docs/upstream/crispasr_triage.md` (PLAN §6).
- **Upstream first** (L16). Before porting a family or building anything generic, check upstream:
  `git ls-tree -d --name-only upstream/main src/models/ src/community_models/`

### Upstream sync rules (formerly tracker Operating Rules 6–7)

1. **An audio.cpp sync ends in a recorded merge, never a content copy.**
   - Close with a real merge, or `git merge -s ours upstream/main` carrying a per-commit disposition ledger in the message.
   - Audit each commit by content: grep the symbols, run `git format-patch -1 --stdout <sha> | git apply --check -`, and confirm the symbols the patch *calls* exist.
   - Verify with `git rev-list --left-right --count HEAD...upstream/main`; the right side must be `0`.
   - **Never `git pull` this repo.** `git fetch upstream` is always safe.
   - Upstream can rewrite a release commit after we merged it (`9bdd1d90` → `4d88768f`); disposition it as a re-release.
2. **transcribe.cpp:** pull the sibling, then record one row per commit in `docs/upstream/transcribe_cpp_triage.md`, and advance its `` Triage watermark: `<sha>` `` line.
   - **That line is parsed by `scripts/sync-deps.sh`. Do not move the file or reword the line.**
   - Never adopt a commit because of its subject line: `6c767184` ("VAD…") also broke CPU Whisper.
   - When a gate fails, run the same test on the parent before blaming anything (LESSONS A5).
3. **An audio.cpp merge is a ggml event.** audio.cpp vendors its own hand-edited ggml, so its changes arrive as untracked edits inside the merge. Run `scripts/sync-ggml.sh --check` after every merge, and turn every delta it reports into a patch before the next sync.

### Dependency sync routine (at every phase boundary and before any release)

```bash
scripts/sync-deps.sh                # drift on all sources (fetches remote refs; never touches the tree)
scripts/sync-deps.sh --fetch        # + fast-forward the sibling reference repos
scripts/sync-deps.sh --verify-ggml  # + prove external/ggml == pin + tracked patches
```

**ggml** is vendored and **generated** at `external/ggml/`, pinned in `external/ggml/UPSTREAM`.
- Never hand-edit it. Every downstream delta lives in `patches/ggml/NNNN-*.patch`, applied in filename order by `scripts/sync-ggml.sh`.
- Keep our pin **at or above** transcribe.cpp's `ggml/UPSTREAM` sha. `status.sh` shows both.

```bash
scripts/sync-ggml.sh master --dry-run   # preview + patch-stack check
scripts/sync-ggml.sh <full-40-char-sha> # a short SHA is not a fetchable ref
scripts/sync-ggml.sh --check            # tree == pin + patches?  (exit 1 = untracked delta)
```

- A patch that breaks on a bump is normal. Rebase it, regenerate it with `git diff --relative=external/ggml`, keep its prose header, and re-run the sync until it is clean.
- Patches are pure LF, while ggml files are mixed CRLF/LF. Never `sed -i` a ggml file.
- To port an audio.cpp ggml commit, use `scripts/port-ggml-commit.sh` (its header has the checklist: fork-only ops, op tables, op_params slots, caller signatures).

## Build and test (Windows; PowerShell; VS 18 via `build_env.bat`)

```powershell
.\build_env.bat cmake --build build-cpu-core --config Release -j 12
& "C:\Program Files\CMake\bin\ctest.exe" --test-dir build-cpu-core -C Release -j 8 --output-on-failure
```

- Call `ctest.exe` **directly**. `build_env.bat`'s `%*` mangles the `|` in `-R "a|b"`.
- Finish a ctest run before relinking: a running test binary causes LNK1104.

| Tree | Configuration | Use |
|---|---|---|
| `build-cpu-core` | `MODEL_SET=core` + UNIFIED_ABI + TRANSCRIBE_ARCHES + tests | Suite of record. Links **no** engine model families. |
| `build-cpu-asr-abi` | `-DAUDIOCPP_MODEL_SET=asr -DENGINE_BUILD_TESTS=ON -DSPEECHCPP_ENABLE_UNIFIED_ABI=ON -DSPEECHCPP_ENABLE_TRANSCRIBE_ARCHES=ON -DAUDIOCPP_BUILD_SERVER=OFF`, Ninja Release | Model-backed C-ABI gates, `ctest -L verdict`, parity tests, WER gates. |
| `build-cuda-core` | `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=89-real` | **Only** for CUDA-specific changes (`.cu`, ggml-cuda patches, CUDA CMake). Routine testing is CPU only. |

- **Benchmarks and port acceptance (R10, [`docs/benchmarking.md`](docs/benchmarking.md)):**
  - Every port or numeric change is measured against **both parents** (audio.cpp and transcribe.cpp) on the same GGUF, machine, backend and threads, using WER/CER on a FLEURS multilingual subset, RTF and peak memory. **It is never accepted if it is slower or less accurate.**
  - Runs: **≥ 3; run 1 is the warm-up** (report it, never average it). Take the mean of runs 2..N: N = 3 while developing, 5 to accept, never more than 10. Interleave the arms.
  - **Build and test on CPU while developing** (the fastest build loop). Before accepting, also run CUDA, Vulkan on NVIDIA and Vulkan on Intel.
  - Keep suites small: one shipped quant, the `core6` languages the model supports, and 10 utterances each in development.
- **Test tiers** (PLAN S1.9):
  - **quick** is the default. Every model-backed verdict or parity test loads the model and runs **one short clip** (`assets/asr_validation/quick/`).
    - Whole tier: `ctest --test-dir build-cpu-asr-abi -C Release -L quick -j 2` (~4 min).
  - **full** is opt-in: configure with `-DSPEECHCPP_FULL_ASR_TESTS=ON`, then run `ctest -L full` (4-clip corpus, batched passes, all modes). Run it before merges and releases, not in the edit loop.
  - **A new model-backed test must follow the same split.** One short clip by default; everything else goes in `*_full`.
- Test models are fetched by `uv run scripts/fetch_asr_test_model.py [--only <substr>]`. It uses a sha256-pinned table and deletes mismatches. `models/` is gitignored.
- **A skip is not a pass.** Missing models must return the ctest skip code.
- **"Environment/asset issue" has been the wrong diagnosis 5 out of 5 times here.** Reproduce and root-cause before accepting it (LESSONS A2).
- `audiocpp_cli`, the server and the WebUI are DEV-profile tools (keep them working). The BUNDLE build is PLAN phase E. Until it exists, the closest is the `client-*` presets, which still ship the legacy `audiocpp` ABI + CLI.

### Post-sync verification (mandatory)

```powershell
.\build_env.bat cmake --build build-cpu-core --config Release -j 12
& "C:\Program Files\CMake\bin\ctest.exe" --test-dir build-cpu-core -C Release -j 8 --output-on-failure
cmake -DSRC_DIR=src/runtime -P tests/lint_teardown.cmake
```

The lint scope is `src/runtime` until the `safe_*` teardown retrofit of the audio.cpp-inherited sessions (sub-task 0.J) lands. It expires with Phase 11c, when `src/runtime` is deleted.

## Python

- **Always `uv run`**. Never bare `python`, `python3` or `pip`. Use `uv pip` / `uv sync`.
- Per-family reference environments: `uv run --project scripts/envs/<family> scripts/<script>.py …`
- **Heredoc trap:** on this machine, heredoc-piped Python loses one backslash level (`'\\0'` became a NUL byte in a source file). Write edit scripts to a file first, and scan edited files for control bytes afterwards (LESSONS F1).

## Formatting

- **Known gap:** `.github/workflows/clang-format.yml` calls `scripts/ci/clang-format.sh`, which was never vendored, so that job cannot pass (PLAN S1.5).
- Until then, format by hand to the style of the file you are editing. Engine code is 2-space LLVM-like; `src/runtime` uses transcribe.cpp's 4-space style.
- Never mass-reformat. Never format vendored trees (`external/ggml/`, `src/runtime/third_party/`, verbatim upstream copies).

## C ABI discipline

**`speech.h` is the target and only first-party ABI (V7-D1).**
- `capi/include/audiocpp.h` is frozen: no new entry points. It is retired in PLAN S3.3.
- `include/audiocpp.h` is **upstream's** file. Leave it exactly as upstream ships it.
- `transcribe.h` becomes a shim.

**No C++ exception may escape a public entry point:**
- Route through an `api_guard_*` wrapper, or be nothrow by construction. Device and registry queries count as calls; guard them.
- Ownership out-params: a non-OK return means `*out == NULL` and nothing leaked.
- Teardown in library code uses `transcribe::safe_*` (`src/runtime/transcribe-backend.h`), never raw `ggml_backend_*_free`. `tests/lint_teardown.cmake` enforces this.
- Host log callbacks are contained at the emission site (`transcribe_log_invoke`).
- `TRANSCRIBE_TEST_*` fault hooks intentionally ship in release artifacts.

## Verification and porting

- Numerical checks: `uv run scripts/validate.py all --family <f> [--variant <v>]`
- Metadata preflight: `uv run scripts/preflight.py --family <f> [--variant <v>]`
- Tensor debugging: `uv run scripts/compare_tensors.py …`
- Never suppress a failing test without a root cause.
- **Porting a model:**
  - Follow `docs/porting/` stages 1–8 (intake → oracle → convert → cpp → quants → bench → WER → ship).
  - The PLAN 11b per-family procedure adds the upstream-first check, the C-ABI + run_batch measurement, and the §5 architecture gates.
  - **Port bar:** WER equal to the arch baseline, not the structural 10 % bound.
  - Before calling a port done, run one clip twice on one session (LESSONS D1).
