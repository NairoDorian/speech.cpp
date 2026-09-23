# speech.cpp Agent Conventions & Fusion Protocol

Read `MULTI_AGENT_FUSION_PLAN_AND_TRACKER.md` and `FUSION_ROADMAP_PLAN.md` for the master fusion roadmap, step-by-step phases, and multi-agent coordination protocol.
Read `CONTRIBUTING.md` for contribution policy, review gates, and coding style. This file is the local command/automation convention sheet for coding agents.

## Fusion Execution Directives
- **Active Repo**: Work in `speech.cpp` **only**.
- **Protocol**: Execute one phase at a time. Run full verification (`build_env.bat ctest`) at the end of each phase.
- **Update Documentation**: Update `CHANGELOG.md`, `progress.md`, and `MULTI_AGENT_FUSION_PLAN_AND_TRACKER.md` at each phase boundary.
- **Pause Rule**: Pause and request user review before beginning the next phase.


## Dual Parentage — transcribe.cpp is a co-parent, not a donor

**`speech.cpp` is equally a child of `audio.cpp` and of `transcribe.cpp`.**
We forked `audio.cpp` for convenience — it gave us the bigger tree to start
from — but that is a *mechanical* accident of how the repo was created, not a
statement about authority. **An improvement landing in `NairoDorian/transcribe.cpp`
is exactly as authoritative as one landing in `0xShug0/audio.cpp`, and must be
tracked, triaged and adopted with the same seriousness.**

Consequences that are easy to get wrong:

- Do **not** describe `transcribe.cpp` as a "merge source", "donor" or
  "read-only reference" and then treat its commits as optional. It is upstream.
  Both parents get the same audit-by-content and the same disposition ledger.
- A dependency bump on **either** parent (ggml, a vendored third-party tree, a
  toolchain pin) is a first-class upstream change for us. When
  `transcribe.cpp` moved ggml to `36da5713` (v0.22.0), that was **our** ggml
  floor moving — not a curiosity to note and defer.
- When the two parents disagree, that is a real design decision to be recorded
  (FUSION_ROADMAP_PLAN / V6 plan), not a tie broken by "audio.cpp is the fork
  base".
- `git` only knows about the `upstream` remote (`0xShug0/audio.cpp`), because
  that is the fork base. The absence of a transcribe.cpp merge-base is a
  **tooling limitation, not a hierarchy** — track it by hand.

## Dependency Sync Routine (run before any release, and regularly)

`speech.cpp` must not go into a release state on stale dependencies. Refresh
**all three** sources, in this order, then verify:

```bash
scripts/sync-deps.sh                # drift on all three (fetches remote REFS; never touches the tree)
scripts/sync-deps.sh --fetch        # + fast-forward the sibling reference repos
scripts/sync-deps.sh --verify-ggml  # + prove external/ggml == pin + tracked patches
```

1. **audio.cpp** (parent, `upstream` remote) — `git fetch upstream`, then
   audit `HEAD..upstream/main` by content and close with a recorded merge.
   Never `git pull` this repo. See Operating Rule 6 in the tracker.
2. **transcribe.cpp** (parent, no git remote here — the sibling checkout
   `../transcribe.cpp` tracks `NairoDorian/transcribe.cpp`) — pull the sibling,
   then triage its new commits against this tree by hand and record one
   disposition row per commit in `docs/upstream/transcribe_cpp_triage.md`. Its
   `Triage watermark:` line is our stand-in for a merge-base; `sync-deps.sh`
   counts the commits past it. Never take a parent commit wholesale because of
   its subject line: `6c767184` ("VAD") also flips Whisper onto a CPU decode
   path that produces garbage.
3. **ggml** (vendored at `external/ggml/`, pinned in `external/ggml/UPSTREAM`) —
   ```bash
   scripts/sync-ggml.sh master --dry-run   # preview + patch-stack check
   scripts/sync-ggml.sh <full-40-char-sha>
   ```
   A **short SHA is not a fetchable ref** — pass the full 40 characters, or a
   branch/tag. Keep our pin at or above transcribe.cpp's `ggml/UPSTREAM` sha.
   **Before any sync, and after every audio.cpp merge**, run
   `scripts/sync-ggml.sh --check`: audio.cpp vendors its own hand-edited ggml,
   so its ggml changes only ever arrive as untracked edits inside a merge.
   Anything `--check` reports must become a `patches/ggml/NNNN-*.patch` first,
   or the next sync deletes it silently. To port an audio.cpp ggml commit
   onto the stack use `scripts/port-ggml-commit.sh` (its header has the
   setup and the post-merge checklist: fork-only ops, op tables, op_params
   slots, caller signatures).

Post-sync verification is mandatory:

```bash
.\build_env.bat cmake --build build-cpu-core --config Release -j 8
.\build_env.bat ctest --test-dir build-cpu-core --output-on-failure -C Release
cmake -DSRC_DIR=src/runtime -P tests/lint_teardown.cmake
```

Note the `src/runtime` scope: unlike transcribe.cpp, whose whole tree is clean,
speech.cpp still has raw ggml teardown calls in the audio.cpp-inherited model
sessions under `src/models/`. Widening this to `src/` fails by design until
Phase 0 sub-task 0.J (the `safe_*` teardown retrofit) lands.

`external/ggml/` is **generated**. Never hand-edit it — every downstream delta
lives in `patches/ggml/NNNN-*.patch` and is re-applied in filename order by
`scripts/sync-ggml.sh`. A ggml bump that breaks a patch is normal: rebase the
patch, regenerate it with `git diff --relative=external/ggml`, keep the prose
header, and re-run the sync until it is clean.

## Python

- ALWAYS use `uv run` for every Python invocation. Never bare `python`,
  `python3`, or `pip`.
- Use `uv pip` for packages, `uv sync` for envs, and `uv run` for scripts.
- Per-family reference environments live under `scripts/envs/<family>/` and are
  invoked as:

```bash
uv run --project scripts/envs/<family> scripts/<script>.py ...
```

## Build

- After C++ changes, build and test the suite of record (from PowerShell; the
  bat wrapper loads the VS 18 x64 environment):

```powershell
.\build_env.bat cmake --build build-cpu-core --config Release -j 12
& "C:\Program Files\CMake\bin\ctest.exe" --test-dir build-cpu-core -C Release -j 8
```

  Call `ctest.exe` directly rather than through `build_env.bat`: the bat's
  `%*` mangles the `|` in a `-R "a|b"` regex.

- `build-cpu-core` links **no** engine models (`MODEL_SET=core`), so a family whose
  transcribe.cpp arch has been retired (moonshine, moonshine_streaming, whisper, ...)
  has no C-ABI path there and its C-ABI gates are not even registered. Run those in
  `build-cpu-asr-abi` (`-DAUDIOCPP_MODEL_SET=asr -DENGINE_BUILD_TESTS=ON
  -DSPEECHCPP_ENABLE_UNIFIED_ABI=ON -DSPEECHCPP_ENABLE_TRANSCRIBE_ARCHES=ON
  -DAUDIOCPP_BUILD_SERVER=OFF`, Ninja Release), same build / ctest commands.
  Routine testing is CPU only; rebuild `build-cuda-core` only for CUDA-specific changes.

- The CLI is `audiocpp_cli` (engine API). transcribe.cpp's `transcribe-cli`
  is not built here. CUDA changes: `build-cuda-core` (`GGML_CUDA=ON`,
  `CMAKE_CUDA_ARCHITECTURES=89-real`).

## Formatting

- **Known gap (verified 2026-09-23):** transcribe.cpp's pinned formatter,
  `scripts/ci/clang-format.sh` (fetched via `uvx`), and its `.clang-format`
  were never vendored here — yet `.github/workflows/clang-format.yml` still
  calls the script, so that CI job cannot pass. Until a formatting policy is
  decided for the two inherited styles (engine code under `src/framework` /
  `src/models` is 2-space LLVM-like; `src/runtime` is transcribe.cpp's 4-space
  style), **format by hand to the style of the file you are editing** and do
  not mass-reformat.
- Whatever policy lands, vendored trees (`external/ggml/`,
  `src/runtime/third_party/`) and verbatim upstream copies
  (`src/runtime/transcribe-unicode-data.cpp`) are never formatted.

## C ABI Exception Discipline

No C++ exception may escape a public entry point.

- A new public entry point must either route through an `api_guard_*`
  wrapper (`src/runtime/transcribe.cpp`) or be nothrow by construction. Device and
  registry queries are not pure reads; guard them.
- Entry points with ownership out-params enforce "non-OK => `*out == NULL`,
  nothing leaked" in their forwarders on every error return.
- Teardown never uses raw `ggml_backend_free` / `ggml_backend_buffer_free` /
  `ggml_backend_sched_free` in library code: use `transcribe::safe_*` from
  `src/runtime/transcribe-backend.h`. `tests/lint_teardown.cmake` fails CI on
  violations.
- Host log callbacks are contained at the emission site
  (`transcribe_log_invoke`).
- The `TRANSCRIBE_TEST_*` fault hooks (`_DEV_INIT_THROW`, `_TEARDOWN_THROW`)
  intentionally ship in release artifacts for wheel clean-install CI.
  Present-but-empty values are inert.

## Verification

- End-to-end numerical checks:

```bash
uv run scripts/validate.py all --family <f> [--variant <v>]
```

`--variant` is required when the family has multiple manifests.

- Manual tensor debugging:

```bash
uv run scripts/compare_tensors.py ...
```

- Cheap metadata/config gates before expensive numerical work:

```bash
uv run scripts/preflight.py --family <f> [--variant <v>]
```

- Never suppress test failures without root cause analysis.

## Porting a New Model

Follow the staged guides in `docs/porting/` (`0-porting.md` onward; the
`porting-*` skills transcribe.cpp ships under `.claude/skills/` are not vendored
here). Stages are independent
and run in order:

```text
porting-1-intake -> porting-2-oracle -> porting-3-convert -> porting-4-cpp
-> porting-5-quants -> porting-6-bench -> porting-7-wer -> porting-8-ship
```

## Git Hygiene

- Do not commit, push, create pull requests, or comment on pull requests unless
  the user explicitly asks for that action.
- Do not reformat unrelated code in the same change as a behavior fix.
