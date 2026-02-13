---
name: dev-environment
description: CMake build system, Git workflow, compiler settings. Use for build issues, CI/CD, or commit/branch conventions.
model: opus
memory: project
---

# Dev Environment Agent

Owns the CMake build system, Git workflow, and development tooling for MPS_DAWN. Build commands are in CLAUDE.md.

> **Skills**: `/new-module` (CMake template), `/add-dep` (add third-party dependency)

## Build Output Structure

```
build/bin/{x64,wasm}/{Debug,Release}/   # Executables
build/lib/x64/{Debug,Release}/          # Static libraries (.lib, .a)
```

## Third-Party Dependencies

Current: **Dawn** (native), **GLFW** (native), **GLM** (all) — git submodules in `third_party/`.

## Git Workflow

### Commit Format

```
<type>(<scope>): <short description>
```

Types: `feat` | `fix` | `refactor` | `docs` | `style` | `test` | `chore`
Scope (optional): `core_util` | `core_platform` | `core_gpu` | *(omit for project-wide)*

Examples: `feat(core_platform): add input manager with keyboard and mouse support`, `chore: add GLFW as git submodule`

### Branch Naming

`main` | `feature/<name>` | `fix/<name>` | `refactor/<name>`

## Testing

No test framework yet. Future structure:

```
tests/
├── core_util/     (logger_test.cpp, math_test.cpp)
└── core_platform/ (window_test.cpp)
```
