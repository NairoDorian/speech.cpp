#!/usr/bin/env bash
# scripts/status.sh — generated project status (PLAN.md §2 / V7-D6).
#
# Read-only: prints measured facts, never edits the tree. Paste its output into
# a handover or CHANGELOG entry instead of hand-writing counts.
#
#   scripts/status.sh                # all sections
#   scripts/status.sh --no-ctest     # skip the ctest -N registration counts
#
# Each line is "key: value" so two runs can be diffed.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
WS="$(dirname "$REPO_ROOT")"
RUN_CTEST=1
[ "${1:-}" = "--no-ctest" ] && RUN_CTEST=0

loc() { find "$@" \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' \) -print0 2>/dev/null | xargs -0 cat 2>/dev/null | wc -l | tr -d ' '; }

echo "== git"
echo "head: $(git rev-parse --short HEAD) $(git log -1 --format=%ad --date=short)"
echo "dirty_paths: $(git status --porcelain | wc -l | tr -d ' ')"
if git rev-parse -q --verify upstream/main >/dev/null; then
  echo "audio_cpp_ahead_behind: $(git rev-list --left-right --count HEAD...upstream/main | tr '\t' '/')"
fi
echo "stashes: $(git stash list | wc -l | tr -d ' ')"

echo "== parents"
WM=$(sed -n 's/^Triage watermark: `\([0-9a-f]*\)`.*/\1/p' docs/upstream/transcribe_cpp_triage.md | head -1)
if [ -d "$WS/transcribe.cpp/.git" ] && [ -n "$WM" ]; then
  echo "transcribe_cpp_watermark: ${WM:0:8}"
  echo "transcribe_cpp_untriaged: $(git -C "$WS/transcribe.cpp" rev-list --count "$WM"..HEAD 2>/dev/null)"
fi
CW=$(sed -n 's/^watermark: \([0-9a-f]*\).*/\1/p' docs/upstream/crispasr_triage.md 2>/dev/null | head -1)
if [ -d "$WS/CrispASR/.git" ] && [ -n "$CW" ]; then
  echo "crispasr_watermark: $CW"
  echo "crispasr_commits_since: $(git -C "$WS/CrispASR" rev-list --count "$CW"..HEAD 2>/dev/null)"
fi

echo "== ggml"
echo "pin: $(sed -n 's/^sha: *//p' external/ggml/UPSTREAM | head -1 | cut -c1-8) (our UPSTREAM has no version field; resolve with git -C <ggml clone> describe)"
echo "patches: $(ls patches/ggml/*.patch 2>/dev/null | wc -l | tr -d ' ')"
if [ -f "$WS/transcribe.cpp/ggml/UPSTREAM" ]; then
  echo "transcribe_cpp_floor: $(sed -n 's/^sha: *//p' "$WS/transcribe.cpp/ggml/UPSTREAM" | head -1 | cut -c1-8) $(sed -n 's/^version: *//p' "$WS/transcribe.cpp/ggml/UPSTREAM" | head -1)"
fi

echo "== size (C/C++ LOC)"
echo "src_models: $(loc src/models)  dirs: $(ls -d src/models/*/ | wc -l | tr -d ' ')"
echo "src_community_models: $(loc src/community_models)  dirs: $(ls -d src/community_models/*/ | wc -l | tr -d ' ')"
echo "src_framework: $(loc src/framework)"
echo "src_runtime: $(loc src/runtime)"
echo "runtime_arch_dirs: $(ls -d src/runtime/arch/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
echo "cmakelists_lines: $(wc -l < CMakeLists.txt | tr -d ' ')"

echo "== architecture gates (PLAN §5; should trend to 0 / allow-list)"
echo "private_kvcache_structs: $(grep -rlE 'struct +[A-Za-z:]*KvCache *\{' include/engine/models src/models src/community_models 2>/dev/null | tr '\n' ' ')"
echo "sched_new_outside_framework: $(grep -rl ggml_backend_sched_new src/models src/community_models 2>/dev/null | wc -l | tr -d ' ')"
echo "getenv_names_first_party: $(grep -rhoE 'getenv\("[A-Za-z0-9_]+"' src app capi 2>/dev/null | sort -u | wc -l | tr -d ' ')"
echo "c_abi_headers: $(ls capi/include/audiocpp.h include/audiocpp.h include/transcribe/transcribe.h include/speech/speech.h 2>/dev/null | tr '\n' ' ')"

echo "== docs"
echo "root_md_kb: $(du -ck ./*.md 2>/dev/null | tail -1 | cut -f1)"

if [ "$RUN_CTEST" = 1 ]; then
  echo "== ctest registrations (not results)"
  CTEST="ctest"
  [ -x "/c/Program Files/CMake/bin/ctest.exe" ] && CTEST="/c/Program Files/CMake/bin/ctest.exe"
  for t in build-cpu-core build-cpu-asr-abi; do
    [ -d "$t" ] && echo "$t: $("$CTEST" --test-dir "$t" -N 2>/dev/null | sed -n 's/^Total Tests: *//p')"
  done
fi
