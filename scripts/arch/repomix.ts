// scripts/arch/repomix.ts
//
// Packs the repository (or a part of it) into repomix-output.xml, the file you
// hand to an LLM when it needs file *contents*. For the structure, read
// ARCHITECTURE.md (generated, committed) first: it is small and always current.
//
//   bun repomix.ts                         whole first-party tree (~10M tokens: usually too big)
//   bun repomix.ts src/framework/asr tests/unittests/test_asr*    scoped pack (what you normally want)
//   bun repomix.ts --compress src/models/whisper                  tree-sitter compressed signatures
//
// The output is gitignored: it is a derived copy of the tree and goes stale on
// the next edit. Configuration (exclusions, header) is repomix.config.json at the
// repo root. The repomix version is pinned in package.json (bun.lock).

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

const ARCH = import.meta.dirname;
const REPO = path.resolve(ARCH, "..", "..");
const BIN = path.join(ARCH, "node_modules", "repomix", "bin", "repomix.cjs");

if (!existsSync(BIN)) {
  const r = spawnSync("bun", ["install", "--frozen-lockfile"], { cwd: ARCH, stdio: "inherit" });
  if (r.status !== 0) process.exit(r.status ?? 1);
}

const args = process.argv.slice(2);
const flags = args.filter((a) => a.startsWith("-"));
const paths = args.filter((a) => !a.startsWith("-"));
const cli = ["--config", path.join(REPO, "repomix.config.json"), ...flags];
if (paths.length > 0) {
  // Scoped pack: include only the given paths (dirs become dir/**).
  const inc = paths.map((p) => (existsSync(path.join(REPO, p)) && !path.extname(p) ? `${p.replace(/\/$/, "")}/**` : p));
  cli.push("--include", inc.join(","));
}
const r = spawnSync("bun", [BIN, ...cli], { cwd: REPO, stdio: "inherit" });
process.exit(r.status ?? 1);
