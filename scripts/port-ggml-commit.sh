#!/usr/bin/env bash
# port-ggml-commit.sh — 3-way merge one audio.cpp commit's external/ggml changes
# onto a checkout of our ggml patch stack, file by file, with every side
# LF-normalized.
#
# Why this exists: audio.cpp vendors its own hand-edited ggml and stores those
# files with CRLF, while speech.cpp's stack (pin + patches/ggml/*.patch) is LF.
# `git apply -3` of an audio.cpp ggml diff therefore turns every touched file
# into a single whole-file conflict. Merging each file with `git merge-file`
# on LF-normalized base / theirs / ours leaves only the real conflicts.
#
# Setup (once): a scratch clone of ggml-org/ggml, a commit on the pin with the
# tree LF-normalized, then patches/ggml/*.patch applied as one commit each.
# Make the audio.cpp objects reachable from it:
#   git remote add speech <path-to-speech.cpp>
#   git fetch speech refs/remotes/upstream/main:refs/remotes/audiocpp/main
#
# Usage (run from the root of that ggml checkout):
#   scripts/port-ggml-commit.sh <audio.cpp-sha> [path-ERE]
# The optional ERE selects ggml-relative paths (e.g. to take only the API and
# CPU files: '^(include/ggml\.h|src/ggml\.c|src/ggml-cpu/.*)$').
#
# After resolving the conflict markers, check before committing:
#   1. every GGML_OP_* the port uses exists in include/ggml.h (audio.cpp has
#      fork-only ops, e.g. GGML_OP_IM2COL_FAST_1D);
#   2. GGML_OP_NAME / GGML_OP_SYMBOL line up with the enum, and the
#      static_assert(GGML_OP_COUNT == N) values match;
#   3. op_params slots: newer upstream ggml may already use the slot
#      audio.cpp writes (0.24.0's SSM_SCAN keeps K in slot 0);
#   4. build: engine callers may need signature fixes (ggml_ssm_scan gained K).
# Then emit the patch with a provenance header and verify with
# `scripts/sync-ggml.sh <full-sha> --force && scripts/sync-ggml.sh --check`.
set -u
if [ $# -lt 1 ]; then
  sed -n '2,33p' "$0"
  exit 2
fi
sha="$1"
filter="${2:-.}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
lf() { tr -d '\r'; }
git diff-tree --no-commit-id -r --name-status "$sha^" "$sha" -- external/ggml | while read -r status path rest; do
  rel="${path#external/ggml/}"
  if ! printf "%s" "$rel" | grep -Eq "$filter"; then echo "SKIPPED   $rel"; continue; fi
  case "$status" in
    A)
      mkdir -p "$(dirname "$rel")"
      git show "$sha:$path" | lf > "$rel"
      echo "ADDED     $rel"
      ;;
    D)
      rm -f "$rel"
      echo "DELETED   $rel"
      ;;
    M)
      if [ ! -f "$rel" ]; then
        echo "MISSING   $rel  (modified upstream, absent in ours)"
        continue
      fi
      git show "$sha^:$path" | lf > "$tmp/base"
      git show "$sha:$path"  | lf > "$tmp/theirs"
      tr -d '\r' < "$rel" > "$tmp/ours"
      cp "$tmp/ours" "$rel"
      if git merge-file -L ours -L base -L "audio.cpp@${sha:0:8}" "$rel" "$tmp/base" "$tmp/theirs"; then
        echo "CLEAN     $rel"
      else
        echo "CONFLICT  $rel  ($(grep -c '^<<<<<<< ' "$rel") hunk(s))"
      fi
      ;;
    *)
      echo "UNHANDLED $status $path $rest"
      ;;
  esac
done
