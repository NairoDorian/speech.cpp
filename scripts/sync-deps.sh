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
# AGENTS.md "Upstream sync rules" (formerly tracker Operating Rules 6 and 7).
#
# Usage:
#   scripts/sync-deps.sh                # drift report (fetches remote REFS only)
#   scripts/sync-deps.sh --fetch        # + fast-forward the sibling checkouts
#   scripts/sync-deps.sh --verify-ggml  # + prove external/ggml == pin + patches
#   scripts/sync-deps.sh --offline      # no network: report against cached refs
#   scripts/sync-deps.sh --help
#
# This script NEVER modifies speech.cpp's working tree: it does not pull,
# merge, or re-vendor. It does `git fetch` remote-tracking refs by default,
# because a report computed against stale refs is worse than none: on
# 2026-09-23 it printed "0 behind" while upstream/main was 61 commits ahead.
# It tells you what is stale and prints the exact command to fix each one.
# Adopting upstream changes stays a human/agent decision with an audit trail.
#
# transcribe.cpp has no merge-base here, so its "what have we absorbed" line is
# a hand-maintained watermark: the `Triage watermark:` line in
# docs/upstream/transcribe_cpp_triage.md. Advance it only after every commit up
# to it has a disposition row in that ledger.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIBLINGS="$(cd "${REPO_ROOT}/.." && pwd)"
TRIAGE_LEDGER="${REPO_ROOT}/docs/upstream/transcribe_cpp_triage.md"

DO_FETCH=0
DO_VERIFY_GGML=0
OFFLINE=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --fetch) DO_FETCH=1; shift ;;
        --verify-ggml) DO_VERIFY_GGML=1; shift ;;
        --offline) OFFLINE=1; shift ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
if [ "$OFFLINE" -eq 1 ] && { [ "$DO_FETCH" -eq 1 ] || [ "$DO_VERIFY_GGML" -eq 1 ]; }; then
    echo "--offline cannot be combined with --fetch or --verify-ggml" >&2
    exit 2
fi

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '  \033[33m! %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m %s\n' "$*"; }

STALE=0

# ---------------------------------------------------------------- parent 1/2
bold "[1/3] audio.cpp  (parent — git remote 'upstream')"
if git -C "$REPO_ROOT" remote get-url upstream >/dev/null 2>&1; then
    if [ "$OFFLINE" -eq 0 ]; then
        # Refs only: safe on the WIP repo, and the only way the count is real.
        git -C "$REPO_ROOT" fetch --quiet upstream --prune \
            || warn "fetch of 'upstream' failed — the count below uses cached refs"
    else
        warn "offline: the count below uses cached refs and may be stale"
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
    if [ "$OFFLINE" -eq 0 ]; then
        git -C "$TC" fetch --quiet origin --prune \
            || warn "fetch of transcribe.cpp origin failed — using cached refs"
    fi
    if [ "$DO_FETCH" -eq 1 ]; then
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

    # What WE have absorbed: the ledger's watermark, not the sibling's HEAD.
    WATERMARK=""
    if [ -f "$TRIAGE_LEDGER" ]; then
        WATERMARK="$(sed -n 's/^Triage watermark:[[:space:]]*`\{0,1\}\([0-9a-f]\{7,40\}\).*/\1/p' "$TRIAGE_LEDGER" | head -1)"
    fi
    if [ -z "$WATERMARK" ]; then
        STALE=1
        warn "no 'Triage watermark:' line in ${TRIAGE_LEDGER#"$REPO_ROOT"/}"
    elif ! git -C "$TC" cat-file -e "${WATERMARK}^{commit}" 2>/dev/null; then
        STALE=1
        warn "triage watermark ${WATERMARK} is not a commit in the sibling checkout"
    else
        UNTRIAGED="$(git -C "$TC" rev-list --count --no-merges "${WATERMARK}..origin/main" 2>/dev/null || echo '?')"
        echo "  triaged up to : ${WATERMARK:0:8}  (docs/upstream/transcribe_cpp_triage.md)"
        if [ "$UNTRIAGED" = "0" ]; then
            ok "every transcribe.cpp commit up to origin/main has a disposition"
        else
            STALE=1
            warn "${UNTRIAGED} transcribe.cpp commit(s) past the triage watermark"
            git -C "$TC" log --no-merges --oneline --reverse "${WATERMARK}..origin/main" 2>/dev/null \
                | head -25 | sed 's/^/       /'
            echo "       -> audit each BY CONTENT against this tree, record a disposition row,"
            echo "          then advance the watermark (same discipline as Rule 6)."
        fi
    fi
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

    if [ "$DO_VERIFY_GGML" -eq 1 ]; then
        # The invariant is the patch stack, not the tree. Merges from audio.cpp
        # (which vendors its own hand-edited ggml) are how ~1,800 untracked lines
        # got in before 2026-09-23; only a real pin + patches regeneration shows it.
        if bash "${SCRIPT_DIR}/sync-ggml.sh" "$OUR_SHA" --check >&2; then
            ok "external/ggml == pin + tracked patches (round-trip verified)"
        else
            STALE=1
            warn "external/ggml does NOT equal pin + patches — an untracked delta exists"
            echo "       -> capture it as patches/ggml/NNNN-*.patch before ANY sync (see UPSTREAM notes)"
        fi
    fi

    if [ "$OFFLINE" -eq 0 ] && [ -n "$GGML_REPO" ]; then
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
  cmake -DSRC_DIR=src/runtime -P tests/lint_teardown.cmake   # src/ fails by design (Phase 0 sub-task 0.J)
EOF
echo

if [ "$STALE" -ne 0 ]; then
    bold "RESULT: dependencies are STALE — see the warnings above."
    exit 1
fi
bold "RESULT: all three sources current."
