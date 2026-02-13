---
name: sync-docs
description: Scan the codebase and update all .md documentation files (CLAUDE.md, agents, skills, memory) to reflect the current project state
---

# Sync Documentation Workflow

Audit all project documentation against the actual codebase. Two goals:

1. **Accuracy** — update anything outdated, missing, or inconsistent
2. **Clarity** — restructure and rewrite for readability and efficiency

## Scope

| File / Directory | What to check |
|-----------------|---------------|
| `CLAUDE.md` | Module layers, third-party deps, namespaces, coding conventions, agent table, skills table |
| `.claude/agents/*.md` | Accuracy of descriptions, owned modules, code examples, referenced patterns |
| `.claude/skills/*/SKILL.md` | Code examples match current source, referenced file paths and line numbers still valid |
| `.claude/agent-memory/*/MEMORY.md` | Stale entries about completed/reverted work |

## Steps

### 1. Gather current codebase state

Scan these to build a snapshot of what actually exists:

- `src/*/` — list all module directories
- `src/*/CMakeLists.txt` — actual dependencies and link targets
- `src/CMakeLists.txt` — registered subdirectories
- `CMakeLists.txt` (root) — third-party subdirectories and their platform guards
- `third_party/` — actual submodule directories
- `.gitmodules` — registered submodules
- `.claude/agents/*.md` — all agent definitions
- `.claude/skills/*/SKILL.md` — all skill definitions

### 2. Audit CLAUDE.md

Check each section against the snapshot:

- **Module Layers**: Does the dependency tree match actual `target_link_libraries`? Any new or removed modules?
- **Third-Party Dependencies**: Do listed libs match `third_party/` contents? Platform annotations correct?
- **Namespaces**: Do listed namespace mappings match actual code?
- **Coding Conventions**: Do the rules reflect current practice? Any new patterns not yet documented?
- **Agent Table**: Does each row match the agent .md files? Any new agents or renamed ones?
- **Skills Table**: Does each row match the skill SKILL.md files? Any new skills or removed ones?

### 3. Audit agent definitions (`.claude/agents/*.md`)

For each agent:

- Does `description` accurately reflect its current responsibilities?
- Are owned modules still correct?
- Do code examples compile against current source?
- Are referenced skills up to date?

### 4. Audit skill definitions (`.claude/skills/*/SKILL.md`)

For each skill:

- Do code examples match current codebase patterns (namespaces, naming, types)?
- Are referenced source file paths (`> Source: file:line`) still valid?
- Are code snippets consistent with the `/style` guide?

### 5. Audit agent memory (`.claude/agent-memory/*/MEMORY.md`)

For each memory file:

- Remove entries about work that was reverted or superseded
- Remove entries that are now documented in CLAUDE.md or agent definitions (avoid duplication)
- Keep only actionable, current information

### 6. Improve readability and structure

For every file touched, also evaluate and improve:

- **Redundancy** — merge duplicated information; keep a single source of truth
- **Scannability** — prefer tables and bullet lists over long prose; use headers to divide sections
- **Conciseness** — cut filler words and obvious statements; every sentence should earn its place
- **Logical ordering** — group related items together; order by importance or dependency
- **Consistent tone** — imperative for rules ("Use X"), declarative for descriptions ("Handles Y")
- **Formatting** — consistent heading levels, table alignment, code fence language tags

Example improvements:

```
# Before (wordy)
This agent is responsible for handling all tasks related to the
management of modules in the project, including creating new ones.

# After (concise)
Manages module lifecycle — creation, architecture, coding standards.
```

```
# Before (flat list, hard to scan)
- Use PascalCase for classes
- Use PascalCase for methods
- Use snake_case_ for private members
- Use IPrefix for interfaces

# After (table, scannable)
| Element | Convention | Example |
|---------|-----------|---------|
| Class / Method | PascalCase | `WindowNative`, `Initialize()` |
| Private member | snake_case_ | `window_`, `is_focused_` |
| Interface | IPrefix | `IWindow` |
```

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
- CLAUDE.md — <what changed>
- .claude/agents/foo.md — <what changed>
- ...

No changes needed:
- <files that were already up to date>
```

## Rules

### Accuracy
- **Never remove information that is still accurate** — update or reorganize, don't delete
- **Do not invent information** — only document what actually exists in the codebase
- **Line references in skills**: Update `> Source: file:line` when line numbers have shifted
- **Code examples**: Update to match current codebase patterns (e.g., nested namespaces, `[[nodiscard]]`)

### Clarity
- **Shorten, don't pad** — if the same meaning fits in fewer words, use fewer words
- **Tables over lists** when comparing items with shared attributes (name, convention, example)
- **One concept per section** — split overloaded sections, merge scattered fragments
- **Consistent formatting** across all files — heading levels, code fences, table style

### Process
- **Atomic edits** — make each change individually so it can be reviewed
- **Preserve intent** — restructuring for clarity must not change the meaning
