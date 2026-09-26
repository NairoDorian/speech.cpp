// scripts/arch/pre-commit.ts
//
// The pre-commit routine. `.githooks/pre-commit` runs it once the repo points
// git at that directory:  git config core.hooksPath .githooks
//
// It must stay FAST (seconds, no builds) so it never tempts anyone to bypass it.
// Builds and tests belong to the phase-exit routine in AGENTS.md, not here.
//
//   1. Regenerate ARCHITECTURE.md (repomix-derived map) and stage it if it changed.
//   2. Staged-file hygiene, each check encoding a lesson (LESSONS.md):
//      - control bytes in text files      (F1: heredoc edits once wrote a NUL into C++)
//      - absolute user paths in code      (A11: foreign-machine paths came in with copies)
//      - external/ggml edited without a sync or patch (B3: the tree is generated;
//        hand edits are silently deleted by the next sync)
//
// Bypass (rarely right): git commit --no-verify

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

const ARCH = import.meta.dirname;
const REPO = path.resolve(ARCH, "..", "..");
const TAG = "[pre-commit]";
const t0 = Date.now();

const git = (...a: string[]) => spawnSync("git", a, { cwd: REPO, encoding: "utf8" });
const fail = (msg: string): never => {
  console.error(`${TAG} FAIL: ${msg}\n${TAG} fix it, or bypass with --no-verify if you are sure.`);
  process.exit(1);
};

// ---------------------------------------------------------------- 1. architecture map
if (!existsSync(path.join(ARCH, "node_modules", "repomix"))) {
  const r = spawnSync("bun", ["install", "--frozen-lockfile"], { cwd: ARCH, stdio: "inherit" });
  if (r.status !== 0) fail("could not install scripts/arch dependencies (bun install)");
}
const gen = spawnSync("bun", ["gen-architecture.ts"], { cwd: ARCH, encoding: "utf8" });
if (gen.status !== 0) fail(`ARCHITECTURE.md generation failed:\n${gen.stderr}`);
if (gen.stderr.trim()) console.warn(gen.stderr.trim().split("\n").map((l) => `${TAG} ${l}`).join("\n"));
// Stage only when the working copy differs from the index (or the file is not in the index yet).
const inIndex = git("ls-files", "--error-unmatch", "--", "ARCHITECTURE.md").status === 0;
const changed = !inIndex || git("diff", "--quiet", "--", "ARCHITECTURE.md").status !== 0;
if (changed) {
  git("add", "--", "ARCHITECTURE.md");
  console.log(`${TAG} ARCHITECTURE.md regenerated and staged`);
}

// ---------------------------------------------------------------- 2. staged-file hygiene
const staged = git("diff", "--cached", "--name-only", "--diff-filter=ACMR").stdout.split("\n").filter(Boolean);
const TEXT = /\.(c|cc|cpp|cxx|h|hpp|cu|cuh|metal|cmake|py|sh|ts|js|json|md|txt|toml|yml|yaml|bat|ps1)$|CMakeLists\.txt$/i;
const CODE = /\.(c|cc|cpp|cxx|h|hpp|cu|cuh|metal|cmake|py|sh|ts|js|toml|yml|yaml|bat|ps1)$|CMakeLists\.txt$/i;
const USER_PATH = /(?:\b[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s"'`]+|\/Users\/[^/\s"'`]+|\/home\/[^/\s"'`]+)/;
const EXEMPT = /^(external\/|docs\/archive\/|tests\/golden\/|scripts\/arch\/pre-commit\.ts$)/;
const problems: string[] = [];

for (const f of staged) {
  if (EXEMPT.test(f) || !TEXT.test(f)) continue;
  const full = path.join(REPO, f);
  if (!existsSync(full)) continue;
  const buf = readFileSync(full);
  for (let i = 0; i < buf.length; i++) {
    const c = buf[i]!;
    if (c < 0x20 && c !== 0x09 && c !== 0x0a && c !== 0x0d) {
      problems.push(`${f}: control byte 0x${c.toString(16).padStart(2, "0")} at offset ${i} (LESSONS F1)`);
      break;
    }
  }
  if (CODE.test(f)) {
    const lines = buf.toString("utf8").split("\n");
    const hit = lines.findIndex((l) => USER_PATH.test(l));
    if (hit >= 0) problems.push(`${f}:${hit + 1}: absolute user path "${lines[hit]!.match(USER_PATH)![0]}" (LESSONS A11)`);
  }
}

const ggmlEdits = staged.filter((f) => f.startsWith("external/ggml/") && f !== "external/ggml/UPSTREAM");
if (ggmlEdits.length > 0 && !staged.includes("external/ggml/UPSTREAM") && !staged.some((f) => f.startsWith("patches/ggml/"))) {
  problems.push(`external/ggml is generated but ${ggmlEdits.length} file(s) there are staged with no sync (UPSTREAM) or patches/ggml change (LESSONS B3): ${ggmlEdits.slice(0, 3).join(", ")}${ggmlEdits.length > 3 ? ", …" : ""}`);
}

if (problems.length > 0) fail(`\n  ${problems.join("\n  ")}`);
console.log(`${TAG} ok (${staged.length} staged file(s), ${((Date.now() - t0) / 1000).toFixed(1)} s)`);
