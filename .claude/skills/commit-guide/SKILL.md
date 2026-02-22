---
name: commit-guide
description: Commit message style guide — type selection, format rules, and examples for MPS_DAWN commits
---

# Commit Message Style Guide

## Format

```
<type>(<scope>): <short description>
```

- **type**: required — what kind of change
- **scope**: optional — which module or area is affected
- **short description**: required — imperative mood, lowercase, no period, max ~72 chars

## Types

| Type | When to use | Example |
|------|-------------|---------|
| `feat` | New functionality (class, method, system) | `feat(core_platform): add input manager with keyboard and mouse support` |
| `fix` | Bug fix | `fix(core_util): prevent division by zero in aspect ratio calculation` |
| `refactor` | Code restructuring, no behavior change | `refactor(core_platform): extract GLFW callbacks into separate methods` |
| `style` | Formatting, naming, whitespace only | `style: apply nested namespace syntax across all modules` |
| `docs` | Documentation only (.md files, comments) | `docs: add /style skill with modern C++20 formatting guide` |
| `chore` | Build, config, tooling, dependencies | `chore: add GLFW as git submodule` |
| `test` | Adding or updating tests | `test(core_util): add timer precision unit tests` |

### Choosing the right type

- Changed behavior → `feat` or `fix`
- Same behavior, different code → `refactor`
- Only .md / comment changes → `docs`
- Only whitespace / formatting → `style`
- CMake, git, CI, submodules → `chore`

## Scope

Use the module name without `mps::` prefix:

| Scope | When |
|-------|------|
| `core_util` | types, logger, timer, math |
| `core_platform` | window, input |
| `core_gpu` | WebGPU / Dawn |
| `core_database` | database, transactions |
| `core_render` | rendering pipeline |
| `core_simulate` | simulation, device DB |
| `core_system` | system controller |
| `ext_newton` | Newton-Raphson solver + Newton dynamics terms |
| `ext_pd` | Projective Dynamics solver + PD constraint terms |
| `ext_dynamics` | shared constraint types + constraint builder |
| `ext_mesh` | mesh rendering + normal computation |
| `ext_sample` | sample extension (reference) |
| *(omit)* | project-wide or non-module changes |

Multiple modules affected → omit scope, mention in description.

## Description Rules

1. **Imperative mood** — "add", "fix", "remove", not "added", "fixes", "removed"
2. **Lowercase** — no capital first letter
3. **No trailing period**
4. **What, not how** — describe the result, not the implementation
5. **Max ~72 characters** in the subject line

```
# Good
feat(core_platform): add fullscreen toggle with monitor detection
fix(core_util): correct log level comparison in debug filter
refactor: extract GLFW event handling into dedicated callback methods
chore: add glm as third-party submodule

# Bad
feat(core_platform): Added fullscreen toggle.     # past tense, period
fix: Fix bug                                       # vague, capitalized
refactor(core_platform): Refactored the window     # redundant "refactored"
update stuff                                       # no type, meaningless
```

## Multi-line Body (optional)

For complex changes, add a body separated by a blank line:

```
feat(core_gpu): add WebGPU device initialization

Initialize WebGPU adapter and device with error handling.
Configure default limits and required features for Dawn backend.
```

Body rules:
- Blank line between subject and body
- Wrap at 72 characters
- Explain **why**, not what (the diff shows what)
- Use bullet points for multiple items if needed

## Breaking Changes

Prefix the description with `!` when the change breaks existing APIs:

```
feat(core_platform)!: replace raw GLFW handle with opaque window handle
```
