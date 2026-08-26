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
scripts/sync-deps.sh              # report drift on all three (read-only)
scripts/sync-deps.sh --fetch      # + fetch/ff the sibling reference repos
```

1. **audio.cpp** (parent, `upstream` remote) — `git fetch upstream`, then
   audit `HEAD..upstream/main` by content and close with a recorded merge.
   Never `git pull` this repo. See Operating Rule 6 in the tracker.
2. **transcribe.cpp** (parent, no git remote here — the sibling checkout
   `../transcribe.cpp` tracks `NairoDorian/transcribe.cpp`) — pull the sibling,
   then triage its new commits against this tree by hand.
3. **ggml** (vendored at `external/ggml/`, pinned in `external/ggml/UPSTREAM`) —
   ```bash
   scripts/sync-ggml.sh master --dry-run   # preview + patch-stack check
   scripts/sync-ggml.sh <full-40-char-sha>
   ```
   A **short SHA is not a fetchable ref** — pass the full 40 characters, or a
   branch/tag. Keep our pin at or above transcribe.cpp's `ggml/UPSTREAM` sha.

Post-sync verification is mandatory:

```bash
.\build_env.bat cmake --build build-cpu-core --config Release -j 8
.\build_env.bat ctest --test-dir build-cpu-core --output-on-failure -C Release
cmake -DSRC_DIR=src -P tests/lint_teardown.cmake
```

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

- After C++ changes, run:

```bash
cmake --build build --target transcribe-cli
```

## Formatting

- Format our C/C++ before committing. The formatter is pinned and fetched via
  `uvx`, so do not rely on a system clang-format:

```bash
scripts/ci/clang-format.sh            # format our tree in place (default)
scripts/ci/clang-format.sh --check    # verify, no changes
```

- Scope is our code only. Vendored trees (`ggml/`, `src/third_party/`) and
  verbatim upstream copies (`src/transcribe-unicode-data.cpp`) are never
  formatted. CI gates our C/C++ in
  `.github/workflows/clang-format.yml`.

## C ABI Exception Discipline

No C++ exception may escape a public entry point.

- A new public entry point must either route through an `api_guard_*`
  wrapper (`src/transcribe.cpp`) or be nothrow by construction. Device and
  registry queries are not pure reads; guard them.
- Entry points with ownership out-params enforce "non-OK => `*out == NULL`,
  nothing leaked" in their forwarders on every error return.
- Teardown never uses raw `ggml_backend_free` / `ggml_backend_buffer_free` /
  `ggml_backend_sched_free` in library code: use `transcribe::safe_*` from
  `src/transcribe-backend.h`. `tests/lint_teardown.cmake` fails CI on
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

Use the `porting-*` skills in `.claude/skills/`. Stage skills are independent
and run in order:

```text
porting-1-intake -> porting-2-oracle -> porting-3-convert -> porting-4-cpp
-> porting-5-quants -> porting-6-bench -> porting-7-wer -> porting-8-ship
```

## Git Hygiene

- Do not commit, push, create pull requests, or comment on pull requests unless
  the user explicitly asks for that action.
- Do not reformat unrelated code in the same change as a behavior fix.
