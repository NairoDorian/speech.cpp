// scripts/arch/gen-architecture.ts
//
// Generates ARCHITECTURE.md: the fast, summarized map of speech.cpp that the
// README (and AGENTS.md) send every reader to first. It is derived from the
// tree with repomix's pack() API, so the counts are measured, never hand-written.
//
//   bun gen-architecture.ts            regenerate ARCHITECTURE.md
//   bun gen-architecture.ts --check    exit 1 if ARCHITECTURE.md is stale
//
// Run from scripts/arch/ (the pre-commit hook does). Counts are rounded to
// three significant figures so the file changes when the structure changes,
// not on every edited line. Descriptions live in DIR_DESCRIPTIONS below: a
// directory without one is listed as "undescribed" and printed as a warning.
// Add a line here whenever you add a top-level area.

import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { loadFileConfig, mergeConfigs, pack, type PackResult } from "repomix";

const REPO = path.resolve(import.meta.dirname, "..", "..");
const OUT = path.join(REPO, "ARCHITECTURE.md");
const CONFIG = path.join(REPO, "repomix.config.json");
const check = process.argv.includes("--check");

// ---------------------------------------------------------------- descriptions
const DIR_DESCRIPTIONS: Record<string, string> = {
  "src": "Library implementation (C++17).",
  "src/framework": "Shared engine layers every family builds on (R5: families must not re-implement these).",
  "src/framework/asr": "ASR runtime layer: EncDecKVCache, decode drivers, sampling, AsrResult/AsrLimits (PLAN S2).",
  "src/framework/assets": "Bundled-asset resolution (asset_paths: never CWD-relative).",
  "src/framework/audio": "Audio front-ends: MelExtractor, fbank, resampling, chunking, STFT.",
  "src/framework/codecs": "Neural audio codecs shared by TTS/S2S families.",
  "src/framework/core": "Backend/device management, tensors, weights, common utilities.",
  "src/framework/decoders": "Shared decoder building blocks.",
  "src/framework/conditioners": "Conditioning modules for generative TTS/music (speaker, text, prompt encoders).",
  "src/framework/debug": "Debug dump helpers.",
  "src/framework/io": "Audio / file I/O and serialization (wav, safetensors, ...).",
  "src/framework/midi": "MIDI output for music-transcription families.",
  "src/framework/model_spec": "Model-spec schema, loading and validation (model_specs/*.json).",
  "src/framework/package_manager": "Model package resolution (spec → files).",
  "src/framework/sampling": "Token sampling (temperature, top-k/p, penalties) for AR decoders.",
  "include/engine/community_models": "Per-family headers for src/community_models.",
  "capi/include": "Legacy audiocpp.h header (frozen).",
  "capi/src": "Legacy audiocpp C ABI implementation.",
  "capi/test": "Ad-hoc scripts from another machine with foreign absolute paths (delete: PLAN S1.2).",
  "src/framework/modules": "Reusable ggml graph modules (attention, conformer, convs, ...).",
  "src/framework/runtime": "Engine contract implementation: registry, sessions, KV cache, run control.",
  "src/framework/text": "Text processing: normalization, G2P helpers.",
  "src/framework/tokenizers": "TokenizerHub: BPE / SentencePiece / tekken / ... shared tokenizers.",
  "src/models": "Core model families as engine packages (graphs + assets + spec + thin session). See the family table.",
  "src/community_models": "Community model families (engine packages; lighter review). See the family table.",
  "src/runtime": "transcribe.cpp runtime + C ABI implementation (transcribe.h) + remaining arch families. Deleted in PLAN phase 11c.",
  "src/runtime/arch": "transcribe.cpp arch families not yet retired (each retires after its verdict; docs/LEDGER.md §2).",
  "src/capi": "C ABI glue on the engine side.",
  "include": "Public and internal headers.",
  "include/engine": "Engine C++ API: framework contracts and per-family headers.",
  "include/engine/framework": "Engine framework headers (runtime contract: IVoiceModelLoader / ILoadedVoiceModel / I*VoiceTaskSession).",
  "include/engine/models": "Per-family headers for src/models.",
  "include/transcribe": "transcribe.h C ABI + family extension headers. The ABI apps embed today; becomes a shim over speech.h (V7-D1).",
  "capi": "Legacy speech.cpp `audiocpp.h` C ABI (frozen; retired after speech.h, V7-D1).",
  "app": "DEV-profile tools (never a library dependency, R1): CLI, server + WebUI, GGUF tool, model manager, workflow.",
  "app/cli": "audiocpp_cli: run any family from the command line (DEV).",
  "app/server": "audiocpp_server: HTTP API + embedded WebUI (DEV).",
  "app/gguf": "audiocpp_gguf: GGUF conversion / inspection tool (DEV).",
  "app/model_manager": "Model download / install manager (DEV).",
  "app/streaming": "Streaming helpers for the DEV apps.",
  "app/workflow": "JSON pipeline / workflow runner (DEV).",
  "app/common": "Code shared by the DEV apps.",
  "webui": "WebUI sources and configs (DEV profile; served by audiocpp_server).",
  "model_specs": "Model package specifications (JSON): variants, files, options, capabilities, licences.",
  "tests": "Unit, parity, verdict and WER gates. Model-backed tests use the quick tier by default (AGENTS.md).",
  "tests/unittests": "Framework + family unit/parity tests (incl. *_engine_arch_parity, abi_arch_engine_verdict).",
  "tests/transcribe": "Tests vendored from transcribe.cpp (some unregistered: PLAN S1.4).",
  "tests/golden": "Golden outputs and reference fixtures.",
  "tests/tolerances": "Per-family numeric tolerance files (L10).",
  "tests/capi": "Legacy audiocpp C ABI tests.",
  "tests/core": "Framework core tests.",
  "tests/perf": "Performance probes.",
  "tests/fixtures": "Test fixtures.",
  "tests/community_models": "Tests for community families.",
  "tools/audiocpp_cli": "CLI path-test harness (run_audiocpp_cli_path_tests.py) (DEV).",
  "tools/community_models": "Community-family tooling (converters, checks).",
  "tools/streaming": "Streaming test tools.",
  "webui/native": "WebUI app sources (SvelteKit) + demo voices (DEV).",
  "webui/configs": "WebUI model catalog / parameter configs.",
  "assets/asr_validation": "ASR fixtures: quick/ (1 clip, default tier) and librispeech/ (4-clip corpus).",
  "assets/framework": "Assets bundled with framework components (e.g. VAD weights).",
  "docs/archive": "Superseded plans and reports (history, not instructions).",
  "docs/maintainers": "Maintainer notes (loader, catalog, model specs).",
  "docs/proposals": "Design proposals.",
  "docs/superpowers": "Upstream agent plans/specs (audio.cpp).",
  "examples/xcode": "Apple/Xcode integration example.",
  "examples/docker": "Docker example.",
  "scripts": "Maintenance scripts: sync-deps.sh, sync-ggml.sh, status.sh, fetch_asr_test_model.py, converters.",
  "scripts/arch": "This generator, the repomix pack script and the pre-commit routine (bun).",
  "cmake": "CMake helper modules (transcribe test registration, ...).",
  "patches": "Downstream patches; patches/ggml is the ggml invariant (sync-ggml.sh --check).",
  "patches/ggml": "ggml patch stack re-applied on every sync, in filename order.",
  "docs": "Documentation. Plan-level docs: LEDGER.md; reports/; upstream/ triage ledgers; porting/ guides.",
  "docs/reports": "Dated reports: checkpoint audit, strategy, CrispASR analysis, validation and performance.",
  "docs/upstream": "Parent triage ledgers (transcribe_cpp_triage.md is parsed by sync-deps.sh).",
  "docs/porting": "Staged porting guides (from transcribe.cpp; read their speech.cpp notes: engine packages, not src/arch).",
  "docs/models": "Model cards for audio.cpp-origin families.",
  "docs/community_models": "Model cards for community families.",
  "docs/build": "Platform build guides.",
  "tools": "Developer tools: CLI path tests, model manager, converters (DEV).",
  "examples": "Example programs (DEV).",
  "bindings": "Language bindings (Rust first is planned: PLAN phase 13).",
  "assets": "Bundled assets and test fixtures (assets/asr_validation/{quick,librispeech}).",
};

// Directories whose children are listed (one level deeper than the default depth 2).
const EXPAND = new Set([
  "src/framework", "include/engine", "app", "patches", "docs", "tests", "scripts",
]);
// Directories whose children are listed only when described; the rest are summarized in one line
// (tests/ alone has ~90 per-family directories).
const DESCRIBED_ONLY = new Set(["tests", "tools", "webui", "assets", "examples", "docs"]);
// Directories whose children are model families: summarized in the family table instead.
const FAMILY_ROOTS: Record<string, string> = {
  "src/models": "core",
  "src/community_models": "community",
  "src/runtime/arch": "transcribe arch",
};

// Hand-picked entry points (existence is checked; a missing one is a warning).
const ENTRY_POINTS: [string, string][] = [
  ["PLAN.md", "North star (end goal + rules R1–R11), current state, phases."],
  ["AGENTS.md", "How to work here: build/test commands, sync routine, test tiers, git rules."],
  ["LESSONS.md", "Every hard-won lesson; headings state the lesson."],
  ["docs/LEDGER.md", "Decisions (V7-D*), family migration state, deletion rows."],
  ["docs/benchmarking.md", "Port acceptance: speed + WER vs both parents, run protocol, backend matrix."],
  ["docs/glossary.md", "Vocabulary (ASR/TTS/streaming/runtime), metrics (WER, CER, RTF, TTFA, DER, SIM, ...) and benchmarks per task."],
  ["CMakeLists.txt", "The build: options, MODEL_SET selection, audiocpp_add_model blocks, tests."],
  ["include/engine/framework/runtime/model.h", "Engine contract: IVoiceModelLoader / ILoadedVoiceModel."],
  ["include/engine/framework/runtime/session.h", "Engine contract: task sessions (offline, batched, streaming)."],
  ["include/transcribe/transcribe.h", "The C ABI apps embed today."],
  ["src/runtime/transcribe-arch-adapter.cpp", "Bridge: C ABI → engine families (deleted in 11c)."],
  ["scripts/status.sh", "Generated project status (never hand-write counts)."],
  ["scripts/sync-deps.sh", "Parent drift report (audio.cpp, transcribe.cpp, ggml, CrispASR)."],
];

// ---------------------------------------------------------------- helpers
const posix = (p: string) => p.replaceAll("\\", "/");
function sig3(n: number): string {
  if (n < 1000) return String(n);
  const units: [number, string][] = [[1e6, "M"], [1e3, "k"]];
  for (const [v, u] of units) {
    if (n >= v) {
      const x = n / v;
      const d = x >= 100 ? 0 : x >= 10 ? 1 : 2;
      return `${Number(x.toFixed(d))}${u}`;
    }
  }
  return String(n);
}
type Agg = { files: number; lines: number; tokens: number };
const add = (m: Map<string, Agg>, k: string, lines: number, tokens: number) => {
  const a = m.get(k) ?? { files: 0, lines: 0, tokens: 0 };
  a.files += 1; a.lines += lines; a.tokens += tokens;
  m.set(k, a);
};

// ---------------------------------------------------------------- measure
const fileConfig = await loadFileConfig(REPO, CONFIG);
const config = mergeConfigs(REPO, fileConfig, {});
config.output.files = false;
config.security.enableSecurityCheck = false;
if (config.output.git) config.output.git.sortByChanges = false;

const result: PackResult = await pack([REPO], config, () => {}, {
  produceOutput: async () => ({ outputForMetrics: "" }) as never,
});

const perFile = new Map<string, { lines: number; tokens: number }>();
for (const f of result.processedFiles) {
  const p = posix(f.path);
  const lines = f.content.length === 0 ? 0 : f.content.split("\n").length;
  perFile.set(p, { lines, tokens: result.fileTokenCounts[f.path] ?? result.fileTokenCounts[p] ?? 0 });
}

const dirs = new Map<string, Agg>();
const families = new Map<string, Agg>(); // key: "<root>|<family>"
let total: Agg = { files: 0, lines: 0, tokens: 0 };
for (const [p, { lines, tokens }] of perFile) {
  total.files++; total.lines += lines; total.tokens += tokens;
  const parts = p.split("/");
  if (parts.length === 1) { add(dirs, "(root files)", lines, tokens); continue; }
  for (let d = 1; d < parts.length; d++) add(dirs, parts.slice(0, d).join("/"), lines, tokens);
  for (const root of Object.keys(FAMILY_ROOTS)) {
    if (p.startsWith(root + "/")) {
      const fam = p.slice(root.length + 1).split("/")[0];
      if (fam && p.slice(root.length + 1).includes("/")) add(families, `${root}|${fam}`, lines, tokens);
    }
  }
}

// ---------------------------------------------------------------- render
const warnings: string[] = [];
const out: string[] = [];
out.push("# speech.cpp — architecture map");
out.push("");
out.push("> **Generated** by `scripts/arch/gen-architecture.ts` from the working tree with [repomix](https://github.com/yamadashy/repomix) (pinned in `scripts/arch/package.json`). **Do not edit by hand.** The pre-commit hook regenerates it (`.githooks/pre-commit`). To change a description, edit `DIR_DESCRIPTIONS` in the generator.");
out.push(">");
out.push("> **Reading order for any agent or new contributor:** this map → [`PLAN.md`](PLAN.md) *North star* (the end goal + rules R1–R11) → [`AGENTS.md`](AGENTS.md) → [`LESSONS.md`](LESSONS.md) headings. For file contents, generate a pack with `bun scripts/arch/repomix.ts [paths…]` (gitignored `repomix-output.xml`).");
out.push("");
out.push(`**Scope measured:** ${total.files} first-party files, ~${sig3(total.lines)} lines, ~${sig3(total.tokens)} tokens (o200k). Excludes \`external/\` (vendored ggml), build trees, models, binaries and \`docs/archive/\`; see \`repomix.config.json\`. Counts are rounded to 3 significant figures.`);
out.push("");
out.push("## 1. Layers");
out.push("");
out.push("```");
out.push("host apps (FreeSpeech desktop #1, FreeSpeech Android #2)      DEV tools: app/ (cli, server+WebUI, gguf, model manager)");
out.push("        │  link in-process (BUNDLE profile)                           │ depend on the library, never the reverse (R1)");
out.push("        ▼                                                             ▼");
out.push("C ABI: include/transcribe/transcribe.h today → include/speech/speech.h (target, V7-D1)   [capi/audiocpp.h: legacy, retiring]");
out.push("        │  src/runtime/ (transcribe runtime + ArchAdapter; deleted in 11c)");
out.push("        ▼");
out.push("engine contract: include/engine/framework/runtime (registry, loaders, task sessions)");
out.push("        ▼");
out.push("src/framework/ shared layers (asr, audio, tokenizers, modules, codecs, runtime, assets)");
out.push("        ▼");
out.push("families: src/models/<f>, src/community_models/<f> (+ model_specs/<f>.json)   [src/runtime/arch/<f>: retiring]");
out.push("        ▼");
out.push("external/ggml (generated: pin + patches/ggml)");
out.push("```");
out.push("");
out.push("## 2. Key entry points");
out.push("");
out.push("| File | Why you open it |");
out.push("|---|---|");
for (const [p, why] of ENTRY_POINTS) {
  if (!existsSync(path.join(REPO, p))) warnings.push(`entry point missing: ${p}`);
  out.push(`| [\`${p}\`](${p}) | ${why} |`);
}
out.push("");
out.push("## 3. Directory map");
out.push("");
out.push("| Directory | Files | Lines | Tokens | What it is |");
out.push("|---|--:|--:|--:|---|");
const shown = [...dirs.keys()].filter((d) => {
  if (d === "(root files)") return true;
  const parts = d.split("/");
  if (parts.length <= 1) return true;
  const parent = parts.slice(0, -1).join("/");
  if (Object.keys(FAMILY_ROOTS).includes(parent)) return false;
  if (parts.length === 2) return !DESCRIBED_ONLY.has(parent) || d in DIR_DESCRIPTIONS;
  return parts.length === 3 && EXPAND.has(parent) && (!DESCRIBED_ONLY.has(parts[0]!) || d in DIR_DESCRIPTIONS);
}).sort((a, b) => (a === "(root files)" ? -1 : b === "(root files)" ? 1 : a.localeCompare(b)));
for (const d of shown) {
  const a = dirs.get(d)!;
  let desc = DIR_DESCRIPTIONS[d];
  if (d === "(root files)") desc = "Top-level docs and build entry points (README, PLAN, AGENTS, LESSONS, CHANGELOG, CMakeLists.txt, presets).";
  if (!desc) { desc = "_undescribed_"; warnings.push(`undescribed directory: ${d}`); }
  const depth = d === "(root files)" ? 0 : d.split("/").length - 1;
  const label = d === "(root files)" ? "_(root files)_" : `${"&nbsp;&nbsp;".repeat(depth)}\`${d}/\``;
  out.push(`| ${label} | ${a.files} | ${sig3(a.lines)} | ${sig3(a.tokens)} | ${desc} |`);
  if (DESCRIBED_ONLY.has(d)) {
    const hidden = [...dirs.keys()].filter((c) => c.startsWith(d + "/") && c.split("/").length === 2 && !(c in DIR_DESCRIPTIONS));
    if (hidden.length > 0) {
      const h = hidden.reduce((acc, c) => { const x = dirs.get(c)!; acc.files += x.files; acc.lines += x.lines; acc.tokens += x.tokens; return acc; }, { files: 0, lines: 0, tokens: 0 });
      const sample = hidden.map((c) => c.split("/")[1]).sort().slice(0, 6).join(", ");
      out.push(`| &nbsp;&nbsp;_${hidden.length} more under \`${d}/\`_ | ${h.files} | ${sig3(h.lines)} | ${sig3(h.tokens)} | e.g. ${sample}${hidden.length > 6 ? ", …" : ""} ${d === "tests" ? "(per-family test dirs)" : "(undescribed)"} |`);
    }
  }
}
out.push("");
out.push("## 4. Model families");
out.push("");
const specs = new Set(
  existsSync(path.join(REPO, "model_specs"))
    ? readdirSync(path.join(REPO, "model_specs")).filter((f) => f.endsWith(".json")).map((f) => f.replace(/\.json$/, ""))
    : [],
);
const famRows = [...families.entries()].map(([k, a]) => {
  const [root, fam] = k.split("|");
  return { root: root!, fam: fam!, a };
}).sort((x, y) => x.fam.localeCompare(y.fam) || x.root.localeCompare(y.root));
const counts: Record<string, number> = {};
for (const r of famRows) counts[FAMILY_ROOTS[r.root]!] = (counts[FAMILY_ROOTS[r.root]!] ?? 0) + 1;
out.push(`${Object.entries(counts).map(([k, v]) => `**${v}** ${k}`).join(" · ")} directories. "spec" = \`model_specs/<family>.json\` exists. transcribe arch rows retire one by one (state: [\`docs/LEDGER.md\`](docs/LEDGER.md) §2). Tiers A/B/C (PLAN SC5) will be added here once decided.`);
out.push("");
out.push("| Family | Kind | Files | Lines | spec |");
out.push("|---|---|--:|--:|:-:|");
for (const r of famRows) {
  out.push(`| \`${r.fam}\` | ${FAMILY_ROOTS[r.root]} | ${r.a.files} | ${sig3(r.a.lines)} | ${specs.has(r.fam) ? "✓" : ""} |`);
}
out.push("");
out.push("## 5. Largest files (by tokens)");
out.push("");
out.push("Big files are where review and merge cost concentrate; think twice before growing them.");
out.push("");
out.push("| File | Lines | Tokens |");
out.push("|---|--:|--:|");
[...perFile.entries()].sort((a, b) => b[1].tokens - a[1].tokens).slice(0, 15).forEach(([p, v]) => {
  out.push(`| \`${p}\` | ${sig3(v.lines)} | ${sig3(v.tokens)} |`);
});
out.push("");
const text = out.join("\n");

for (const w of warnings) console.warn(`[arch] warning: ${w}`);
if (check) {
  const cur = existsSync(OUT) ? readFileSync(OUT, "utf8") : "";
  if (cur !== text) { console.error("[arch] ARCHITECTURE.md is stale — run: cd scripts/arch && bun gen-architecture.ts"); process.exit(1); }
  console.log("[arch] ARCHITECTURE.md is up to date");
} else {
  writeFileSync(OUT, text, "utf8");
  console.log(`[arch] wrote ARCHITECTURE.md (${total.files} files, ~${sig3(total.lines)} lines, ~${sig3(total.tokens)} tokens)`);
}
