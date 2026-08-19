#!/usr/bin/env bash
#
# sync-ggml.sh — re-vendor external/ggml/ from upstream ggml-org/ggml at a given
# ref. Adopted from transcribe.cpp (its scripts/sync-ggml.sh); the vendored tree
# here lives at external/ggml/ and follows the same recipe: the upstream tracked
# tree at the SHA recorded in external/ggml/UPSTREAM, plus the ordered downstream
# patches in patches/ggml/.
#
# What it does:
#   1. Fetches the upstream tracked tree at <ref> (a SHA, tag, or branch).
#   2. Materializes it via `git archive` (tracked files only — no .git, no
#      build cruft), minus the paths in EXCLUDES below.
#   3. Applies patches/ggml/*.patch in filename order.
#   4. Swaps the result into external/ggml and rewrites external/ggml/UPSTREAM.
#
# Usage:
#   scripts/sync-ggml.sh                 # re-vendor the CURRENT pinned SHA (repair / verify)
#   scripts/sync-ggml.sh master          # bump to upstream default-branch HEAD
#   scripts/sync-ggml.sh <sha|tag>       # pin to a specific commit or release tag
#   scripts/sync-ggml.sh <ref> --dry-run # show what would change, write nothing
#
# Flags:
#   --dry-run        Resolve the ref and report the file-level diff; do not touch external/ggml.
#   --force          Proceed even if external/ggml has uncommitted local changes (default: abort).
#   --repo <url>     Override the upstream URL (default: the repo: line in UPSTREAM).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGML_DIR="${REPO_ROOT}/external/ggml"
UPSTREAM_FILE="${GGML_DIR}/UPSTREAM"
PATCH_DIR="${REPO_ROOT}/patches/ggml"

# Upstream paths to drop from the snapshot (relative to the ggml tree root).
# .github = upstream CI (irrelevant to a vendor). .pi = Claude-Code prompt
# metadata shipped upstream; no build value, and its SYSTEM.md entry is a
# symlink that Windows tar cannot materialize, so it is dropped too.
EXCLUDES=( ".github" ".pi" )

shopt -s nullglob
PATCHES=( "${PATCH_DIR}"/*.patch )
shopt -u nullglob

REF=""
REPO=""
DRY_RUN=0
FORCE=0

die() { echo "sync-ggml: $*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --force)   FORCE=1; shift ;;
        --repo)    REPO="${2:-}"; [ -n "$REPO" ] || die "--repo needs a URL"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --*)       die "unknown flag: $1" ;;
        *)         [ -z "$REF" ] || die "more than one ref given ($REF, $1)"; REF="$1"; shift ;;
    esac
done

[ -f "$UPSTREAM_FILE" ] || die "missing $UPSTREAM_FILE — run from a checkout with vendored ggml"

CUR_REPO="$(sed -n 's/^repo:[[:space:]]*//p' "$UPSTREAM_FILE" | head -1)"
CUR_SHA="$(sed -n 's/^sha:[[:space:]]*//p'  "$UPSTREAM_FILE" | head -1)"
[ -n "$CUR_REPO" ] || die "no 'repo:' line in $UPSTREAM_FILE"
[ -n "$CUR_SHA"  ] || die "no 'sha:' line in $UPSTREAM_FILE"

REPO="${REPO:-$CUR_REPO}"
REF="${REF:-$CUR_SHA}"

command -v git >/dev/null || die "git not found"
command -v tar >/dev/null || die "tar not found"

if [ "$DRY_RUN" -eq 0 ] && [ "$FORCE" -eq 0 ]; then
    if [ -n "$(git -C "$REPO_ROOT" status --porcelain -- external/ggml 2>/dev/null)" ]; then
        die "external/ggml/ has uncommitted changes. Commit/stash them, or pass --force to overwrite."
    fi
fi

CLONE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sync-ggml.XXXXXX")"
STAGE_DIR="${REPO_ROOT}/external/ggml.sync-stage.$$"
cleanup() { rm -rf "$CLONE_DIR" "$STAGE_DIR"; }
trap cleanup EXIT

echo "sync-ggml: fetching $REF from $REPO ..."
git init  --quiet "$CLONE_DIR"
git -C "$CLONE_DIR" remote add origin "$REPO"
git -C "$CLONE_DIR" fetch --quiet --depth 1 origin "$REF" \
    || die "could not fetch '$REF' (try a branch/tag, or check the SHA is reachable)"
RESOLVED="$(git -C "$CLONE_DIR" rev-parse "FETCH_HEAD^{commit}")"

mkdir -p "$STAGE_DIR"
# Excludes are passed to tar itself (not just post-extraction rm) because .pi/
# contains a symlink (SYSTEM.md -> gg/SYSTEM.md) that Windows tar cannot
# materialize; skipping it at extraction time avoids the failure entirely.
TAR_EXCLUDES=()
for ex in "${EXCLUDES[@]}"; do
    TAR_EXCLUDES+=( "--exclude=${ex}" )
done
git -C "$CLONE_DIR" archive --format=tar "$RESOLVED" | tar -x "${TAR_EXCLUDES[@]}" -C "$STAGE_DIR"
for ex in "${EXCLUDES[@]}"; do
    rm -rf "${STAGE_DIR:?}/${ex}"
done

# Path of the stage dir relative to REPO_ROOT: git apply --directory expects a
# repo-root-relative prefix, and the stage dir is a sibling of external/ggml.
STAGE_NAME="$(realpath --relative-to="$REPO_ROOT" "$STAGE_DIR")"
for patch in "${PATCHES[@]}"; do
    echo "sync-ggml: applying patches/ggml/$(basename "$patch")"
    # Normalize the patch to LF before applying. The stage is LF (git archive
    # honors eol=lf), and a patch file freshly written on Windows can carry CRs
    # in its hunk lines that make context matching fail -- which is how patch
    # 0006's first sync run died even though the patch was correct. Tracked
    # patches are CR-free in the object store (add-time normalization), so
    # stripping CRs from the working copy is always content-preserving here.
    NORMALIZED_PATCH="${CLONE_DIR}/normalized.patch"
    tr -d '\r' < "$patch" > "$NORMALIZED_PATCH"
    git -C "$REPO_ROOT" apply --check --directory="$STAGE_NAME" "$NORMALIZED_PATCH" \
        || die "patch does not apply: patches/ggml/$(basename "$patch")"
    git -C "$REPO_ROOT" apply --directory="$STAGE_NAME" "$NORMALIZED_PATCH"
done

{
    cat <<EOF
repo: ${REPO}
sha:  ${RESOLVED}
patches:
EOF
    if [ "${#PATCHES[@]}" -eq 0 ]; then
        echo "  (none)"
    else
        for patch in "${PATCHES[@]}"; do
            echo "  patches/ggml/$(basename "$patch")"
        done
    fi
    # Everything from the boilerplate paragraph onwards is carried over from the
    # existing UPSTREAM rather than re-emitted. Below that line the file holds
    # hand-written convergence notes -- which fork deltas were restored and which
    # were deliberately not, the per-op status, the audit recipe, the CRLF
    # caveat -- and only the header above (repo/sha/patch list) is generated.
    # Regenerating the whole file would silently delete all of it, which is the
    # opposite of what a sync is for: the header goes stale on every sync, the
    # notes do not.
    if [ -f "$UPSTREAM_FILE" ] && grep -q '^This directory is generated' "$UPSTREAM_FILE"; then
        echo
        sed -n '/^This directory is generated/,$p' "$UPSTREAM_FILE"
    else
        cat <<'EOF'

This directory is generated from the upstream ggml tree at the SHA above, minus
.github/ and .pi/, with the listed downstream patches applied in order. Do not
edit it by hand. Run scripts/sync-ggml.sh <ref> from the repo root to reproduce
or upgrade it; the script rewrites this file.
EOF
    fi
} > "${STAGE_DIR}/UPSTREAM"

find "$STAGE_DIR" -type f -exec touch {} +

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] would re-vendor: ${CUR_SHA:0:12} -> ${RESOLVED:0:12}"
    if diff -rq "$GGML_DIR" "$STAGE_DIR" >/dev/null 2>&1; then
        echo "[dry-run] no differences — external/ggml already matches this ref."
    else
        echo "[dry-run] differences exist (see git diff -- external/ggml after a real sync)"
    fi
    echo "[dry-run] nothing written."
    exit 0
fi

rm -rf "$GGML_DIR"
mv "$STAGE_DIR" "$GGML_DIR"

CHANGED="$(git -C "$REPO_ROOT" status --porcelain -- external/ggml 2>/dev/null | wc -l | tr -d ' ')"
echo "sync-ggml: ggml re-vendored ${CUR_SHA:0:12} -> ${RESOLVED:0:12}"
echo "sync-ggml: ${CHANGED} path(s) changed under external/ggml/  (review: git diff --stat -- external/ggml)"
