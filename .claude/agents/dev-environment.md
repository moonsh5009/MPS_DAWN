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

Current: **Dawn** (native, includes GLFW), **GLM** (all) — git submodules in `third_party/`.

### Dawn CMake Settings

Key cache variables set in root `CMakeLists.txt`:

| Variable | Value | Purpose |
|----------|-------|---------|
| `DAWN_FETCH_DEPENDENCIES` | ON | Auto-fetch Dawn's transitive deps |
| `DAWN_BUILD_SAMPLES` | OFF | Skip sample programs |
| `DAWN_FORCE_SYSTEM_COMPONENT_LOAD` | ON | Load DLLs from System32 (avoids path mismatch with custom output dirs) |
| `TINT_BUILD_CMD_TOOLS` | OFF | Skip Tint CLI tools |
| `TINT_BUILD_TESTS` | OFF | Skip Tint tests |

## Git Workflow

### Commit Format

```
<type>(<scope>): <short description>
```

Types: `feat` | `fix` | `refactor` | `docs` | `style` | `test` | `chore`
Scope (optional): `core_util` | `core_platform` | `core_gpu` | `core_database` | `core_render` | `core_simulate` | `core_system` | *(omit for project-wide)*

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
