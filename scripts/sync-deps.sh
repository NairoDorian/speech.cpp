#!/usr/bin/env bash
#
# sync-deps.sh — report (and optionally refresh) drift across all THREE sources
# speech.cpp depends on. Run this regularly, and always before a release state.
#
# speech.cpp is equally a child of audio.cpp and of transcribe.cpp. We forked
# audio.cpp for convenience, so only it has a git `upstream` remote and only it
# produces a merge-base — that is a TOOLING limitation, not a hierarchy. An
# improvement in transcribe.cpp is exactly as authoritative as one in
# audio.cpp, and a dependency bump on either parent (ggml above all) is a
# first-class upstream change for us. See AGENTS.md "Dual Parentage" and
# MULTI_AGENT_FUSION_PLAN_AND_TRACKER.md Operating Rules 6 and 7.
#
# Usage:
#   scripts/sync-deps.sh            # read-only drift report (default)
#   scripts/sync-deps.sh --fetch    # + fetch and fast-forward the sibling repos
#   scripts/sync-deps.sh --help
#
# This script NEVER modifies speech.cpp: it does not pull, merge, or re-vendor.
# It tells you what is stale and prints the exact command to fix each one.
# Adopting upstream changes stays a human/agent decision with an audit trail.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIBLINGS="$(cd "${REPO_ROOT}/.." && pwd)"

DO_FETCH=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --fetch) DO_FETCH=1; shift ;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '  \033[33m! %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m %s\n' "$*"; }

STALE=0

# ---------------------------------------------------------------- parent 1/2
bold "[1/3] audio.cpp  (parent — git remote 'upstream')"
if git -C "$REPO_ROOT" remote get-url upstream >/dev/null 2>&1; then
    if [ "$DO_FETCH" -eq 1 ]; then
        git -C "$REPO_ROOT" fetch --quiet upstream --prune || true
    fi
    BEHIND="$(git -C "$REPO_ROOT" rev-list --count HEAD..upstream/main 2>/dev/null || echo '?')"
    AHEAD="$(git -C "$REPO_ROOT" rev-list --count upstream/main..HEAD 2>/dev/null || echo '?')"
    if [ "$BEHIND" = "0" ]; then
        ok "0 behind / ${AHEAD} ahead of upstream/main"
    else
        STALE=1
        warn "${BEHIND} commit(s) behind upstream/main (${AHEAD} ahead)"
        git -C "$REPO_ROOT" log --oneline --reverse HEAD..upstream/main 2>/dev/null | sed 's/^/       /'
        echo "       -> audit each BY CONTENT, then close with a recorded merge (Rule 6):"
        echo "          git fetch upstream && git cherry-pick -x <sha>  ...  && git merge -s ours upstream/main"
    fi
else
    warn "no 'upstream' remote configured"
fi
echo

# ---------------------------------------------------------------- parent 2/2
bold "[2/3] transcribe.cpp  (parent — sibling checkout, no remote here)"
TC="${SIBLINGS}/transcribe.cpp"
if [ -d "$TC/.git" ]; then
    if [ "$DO_FETCH" -eq 1 ]; then
        git -C "$TC" fetch --quiet origin --prune || true
        git -C "$TC" merge --ff-only origin/main >/dev/null 2>&1 || warn "could not fast-forward (local work?)"
    fi
    TC_HEAD="$(git -C "$TC" log --oneline -1 2>/dev/null || echo '?')"
    TC_BEHIND="$(git -C "$TC" rev-list --count HEAD..origin/main 2>/dev/null || echo '?')"
    echo "  checkout HEAD : ${TC_HEAD}"
    if [ "$TC_BEHIND" = "0" ]; then
        ok "sibling checkout is current with origin/main"
    else
        STALE=1
        warn "sibling checkout is ${TC_BEHIND} behind its origin/main — re-run with --fetch"
    fi
    echo "       -> transcribe.cpp drift is NOT tracked by git here. Triage its new"
    echo "          commits against this tree by hand; they are as authoritative as"
    echo "          audio.cpp's. Useful: git -C ../transcribe.cpp log --oneline -20"
else
    warn "sibling checkout not found at ${TC}"
fi
echo

# ------------------------------------------------------------------- vendored
bold "[3/3] ggml  (vendored at external/ggml, pinned in external/ggml/UPSTREAM)"
UPSTREAM_FILE="${REPO_ROOT}/external/ggml/UPSTREAM"
if [ -f "$UPSTREAM_FILE" ]; then
    OUR_SHA="$(sed -n 's/^sha:[[:space:]]*//p' "$UPSTREAM_FILE" | head -1)"
    GGML_REPO="$(sed -n 's/^repo:[[:space:]]*//p' "$UPSTREAM_FILE" | head -1)"
    NPATCH="$(find "${REPO_ROOT}/patches/ggml" -name '*.patch' 2>/dev/null | wc -l | tr -d ' ')"
    echo "  our pin       : ${OUR_SHA:0:12}  (+ ${NPATCH} tracked patches)"

    # transcribe.cpp is a PARENT: its ggml floor is our floor.
    TC_UPSTREAM="${TC}/ggml/UPSTREAM"
    if [ -f "$TC_UPSTREAM" ]; then
        TC_SHA="$(sed -n 's/^sha:[[:space:]]*//p' "$TC_UPSTREAM" | head -1)"
        echo "  transcribe.cpp: ${TC_SHA:0:12}"
        if [ "$TC_SHA" != "$OUR_SHA" ]; then
            STALE=1
            warn "our ggml pin differs from parent transcribe.cpp's — ours must be AT OR ABOVE it"
            echo "       -> scripts/sync-ggml.sh ${TC_SHA}"
        else
            ok "matches parent transcribe.cpp's ggml pin"
        fi
    fi

    if [ "$DO_FETCH" -eq 1 ] && [ -n "$GGML_REPO" ]; then
        HEAD_SHA="$(git ls-remote "$GGML_REPO" HEAD 2>/dev/null | awk '{print $1}')"
        if [ -n "$HEAD_SHA" ]; then
            echo "  upstream HEAD : ${HEAD_SHA:0:12}"
            if [ "$HEAD_SHA" != "$OUR_SHA" ]; then
                warn "upstream ggml has moved"
                echo "       -> scripts/sync-ggml.sh ${HEAD_SHA} --dry-run   # preview + patch-stack check"
            fi
        fi
    fi
    echo "       NOTE: a SHORT sha is not a fetchable ref — always pass the full 40 chars."
    echo "       external/ggml is GENERATED. Never hand-edit it; land deltas as"
    echo "       patches/ggml/NNNN-*.patch. A bump that breaks a patch is normal:"
    echo "       rebase the patch, do not drop it."
else
    warn "no external/ggml/UPSTREAM found"
fi
echo

bold "Post-sync verification (mandatory after adopting anything above)"
cat <<'EOF'
  .\build_env.bat cmake --build build-cpu-core --config Release -j 8
  .\build_env.bat ctest --test-dir build-cpu-core --output-on-failure -C Release
  cmake -DSRC_DIR=src -P tests/lint_teardown.cmake
EOF
echo

if [ "$STALE" -ne 0 ]; then
    bold "RESULT: dependencies are STALE — see the warnings above."
    exit 1
fi
bold "RESULT: all three sources current."
