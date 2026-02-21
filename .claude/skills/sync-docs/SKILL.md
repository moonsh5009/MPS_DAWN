---
name: sync-docs
description: Scan the codebase and update all .md documentation files (CLAUDE.md, agents, docs, skills) to reflect the current project state
---

# Sync Documentation Workflow

Audit all project documentation against the actual codebase. Two goals:

1. **Accuracy** — update anything outdated, missing, or inconsistent
2. **Clarity** — restructure and rewrite for readability and efficiency

## Documentation Layers

| Layer | Location | Content |
|-------|----------|---------|
| **Module reference** | `.claude/docs/*.md` | Source code reference: file trees, types, API signatures, shaders |
| **Agent instructions** | `.claude/agents/*.md` | Task instructions: rules, workflows, guidelines, conventions |
| **Project overview** | `CLAUDE.md` | Architecture, namespaces, coding conventions, agent/skill tables |
| **Skill definitions** | `.claude/skills/*/SKILL.md` | Workflow steps and rules for slash commands |

## Scope

| File / Directory | What to check |
|-----------------|---------------|
| `CLAUDE.md` | Module layers, third-party deps, namespaces, coding conventions, agent table, skills table |
| `.claude/docs/*.md` | File trees, key types, API signatures, shader listings |
| `.claude/agents/*.md` | Description accuracy, task guidelines, workflow rules, doc references |
| `.claude/skills/*/SKILL.md` | Code examples, referenced file paths, consistency with `/style` guide |

## Steps

### 1. Gather current codebase state

Scan these to build a snapshot of what actually exists:

- `src/*/` — list all module directories
- `src/*/CMakeLists.txt` — actual dependencies and link targets
- `src/CMakeLists.txt` — registered subdirectories
- `CMakeLists.txt` (root) — third-party subdirectories and their platform guards
- `third_party/` — actual submodule directories
- `.gitmodules` — registered submodules
- `.claude/docs/*.md` — all doc files
- `.claude/agents/*.md` — all agent definitions
- `.claude/skills/*/SKILL.md` — all skill definitions

### 2. Audit CLAUDE.md

Check each section against the snapshot:

- **Module Layers**: Does the dependency tree match actual `target_link_libraries`? Any new or removed modules?
- **Third-Party Dependencies**: Do listed libs match `third_party/` contents? Platform annotations correct?
- **Namespaces**: Do listed namespace mappings match actual code?
- **Coding Conventions**: Do the rules reflect current practice?
- **Agent Table**: Does each row match the agent .md files? Any new agents or renamed ones?
- **Skills Table**: Does each row match the skill SKILL.md files? Any new skills or removed ones?
- **Module Reference Documentation**: Does it accurately describe the two-layer structure?

### 3. Audit module reference docs (`.claude/docs/*.md`)

Doc files are the **source of truth** for each module's structure, types, and API. For each doc:

- **File tree**: Does it match actual files in `src/module/` (or `extensions/ext_*/`) on disk?
- **Key Types table**: Does it list all public types with correct headers and descriptions?
- **API signatures**: Do they match current header files?
- **Shader listings** (if applicable): Do they match `assets/shaders/` directory contents?
- **Design patterns**: Are they still accurate?

### 4. Audit agent definitions (`.claude/agents/*.md`)

Agent files contain **task instructions only** — no source reference content (file trees, type tables, API signatures). For each agent:

- Does `description` accurately reflect its current responsibilities?
- Are owned modules still correct?
- Does it reference the correct doc file(s) using the **CRITICAL** blockquote format? Expected format:
  `> **CRITICAL**: ALWAYS read \`.claude/docs/<module>.md\` FIRST before any task. This doc contains the complete file tree, types, APIs, and shader references. DO NOT read source files (.h/.cpp) to understand the module — only read source files when you need to edit them.`
- Are task guidelines and rules still accurate?
- Are common task workflows still valid?
- Does it contain any source reference content that should be in docs instead?

### 5. Audit skill definitions (`.claude/skills/*/SKILL.md`)

For each skill:

- Do code examples match current codebase patterns (namespaces, naming, types)?
- Are referenced source file paths still valid?
- Are code snippets consistent with the `/style` guide?

### 6. Improve readability and structure

For every file touched, also evaluate and improve:

- **Redundancy** — merge duplicated information; keep a single source of truth
- **Scannability** — prefer tables and bullet lists over long prose
- **Conciseness** — cut filler words; every sentence should earn its place
- **Logical ordering** — group related items; order by importance or dependency
- **Consistent tone** — imperative for rules ("Use X"), declarative for descriptions ("Handles Y")
- **Formatting** — consistent heading levels, table alignment, code fence language tags

### 7. Apply updates

For each change found:

1. State what is outdated or unclear and why
2. Show the proposed update
3. Apply the edit

### 8. Summary

After all updates, output a summary:

```
## Sync Results

Updated:
- .claude/docs/foo.md — <what changed>
- .claude/agents/bar.md — <what changed>
- CLAUDE.md — <what changed>
- ...

No changes needed:
- <files that were already up to date>
```

## Rules

### Source reference vs task instructions

- **Source reference** (file trees, types, API signatures, shaders) → `.claude/docs/*.md`
- **Task instructions** (rules, conventions, workflows, common tasks) → `.claude/agents/*.md`
- If a source reference change is found, update the doc file — not the agent file
- If a task instruction change is found, update the agent file — not the doc file

### Accuracy

- **Never remove information that is still accurate** — update or reorganize, don't delete
- **Do not invent information** — only document what actually exists in the codebase
- **Code examples**: Update to match current codebase patterns (namespaces, naming, types)

### Clarity

- **Shorten, don't pad** — if the same meaning fits in fewer words, use fewer words
- **Tables over lists** when comparing items with shared attributes
- **One concept per section** — split overloaded sections, merge scattered fragments
- **Consistent formatting** across all files

### Process

- **Atomic edits** — make each change individually so it can be reviewed
- **Preserve intent** — restructuring for clarity must not change the meaning
