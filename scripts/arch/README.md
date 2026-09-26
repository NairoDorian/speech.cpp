# scripts/arch — architecture map, repomix pack, pre-commit

| Command (run in `scripts/arch/`) | What it does |
|---|---|
| `bun gen-architecture.ts` | Regenerates `ARCHITECTURE.md` at the repo root from the working tree, via repomix's `pack()` API. It shows the directory map with measured files, lines and tokens, the model-family table, the largest files and the entry points. |
| `bun gen-architecture.ts --check` | Exits 1 if `ARCHITECTURE.md` is stale. |
| `bun repomix.ts [paths…] [--compress]` | Writes `repomix-output.xml` (gitignored) to hand file contents to an LLM. **Scope it**: the whole tree is ~10M tokens. |
| `bun pre-commit.ts` | The pre-commit routine: regenerate and stage the map, then run the staged-file hygiene checks. |

**One-time setup per clone:**

```sh
cd scripts/arch && bun install --frozen-lockfile     # repomix is pinned in package.json / bun.lock
git config core.hooksPath .githooks                  # enable the pre-commit hook
```

**Rules:**
- `ARCHITECTURE.md` is **generated**. Never edit it by hand. To change what it says about a directory, edit `DIR_DESCRIPTIONS` in `gen-architecture.ts`. A new top-level area without a description shows up as a warning.
- Upgrade repomix deliberately: bump the pin, regenerate, and review the diff.
