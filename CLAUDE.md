# CLAUDE.md

## Language Policy

- **Conversation**: Always respond in Korean (한국어).
- **Code, commits, docs, and guide files**: Always write in English.

## Build Commands

```bash
# Native (first time: git submodule update --init --recursive)
cmake -B build && cmake --build build
build\bin\x64\Debug\mps_dawn.exe     # Windows
./build/bin/x64/Debug/mps_dawn       # Linux/macOS

# WASM (requires Emscripten + ninja)
# Windows: run inside emsdk_env.bat shell
emcmake cmake -B build-wasm/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build-wasm/debug
emcmake cmake -B build-wasm/release -DCMAKE_BUILD_TYPE=Release && cmake --build build-wasm/release
# Output: build-wasm/bin/Debug/mps_dawn.html  or  build-wasm/bin/Release/mps_dawn.html
```

No test framework yet; testing is manual (build and run).

## Architecture

C++20 WebGPU graphics engine using Dawn (native) and Emscripten (WASM). CMake static library modules linked via `mps::module_name` aliases.

### Module Layers (dependencies flow downward)

```
src/main.cpp (executable: mps_dawn)
  ├── core_system     (mps::core_system)   — controller: orchestrates database + simulate + render + extensions
  │     ├── core_database  (mps::core_database)  — host ECS (entities, components, transactions, undo/redo)
  │     ├── core_simulate  (mps::core_simulate)  — device DB, CG solver, IDynamicsTerm + IProjectiveTerm interfaces
  │     └── core_render    (mps::core_render)    — rendering pipeline (camera, passes) + IObjectRenderer interface
  ├── core_gpu       (mps::core_gpu)       — WebGPU abstraction (device, buffers, shaders, textures, compute, builders, RAII handles)
  ├── core_platform  (mps::core_platform)  — window, input
  ├── core_util      (mps::core_util)      — types, logger, timer, math
  ├── ext_newton     (mps::ext_newton)     — Newton-Raphson solver + IDynamicsTerm implementations (spring, area) + built-in inertia/gravity
  ├── ext_pd         (mps::ext_pd)         — Projective Dynamics solver + IProjectiveTerm implementations + Chebyshev Jacobi
  ├── ext_dynamics   (mps::ext_dynamics)   — shared constraint types (SpringEdge, AreaTriangle) + constraint builder + GlobalPhysicsParams singleton
  ├── ext_mesh       (mps::ext_mesh)       — mesh rendering + normal computation
  └── ext_sample     (mps::ext_sample)     — minimal reference extension (not linked in main.cpp)

extensions/ext_newton/     — Newton solver + IDynamicsTerm implementations (spring, area) + built-in inertia/gravity
extensions/ext_pd/         — Projective Dynamics solver + IProjectiveTerm implementations (spring, area)
extensions/ext_dynamics/   — shared constraint types, config components, constraint builder
extensions/ext_mesh/       — mesh factories (grid, OBJ, pin/unpin) + normals + indexed triangle rendering
extensions/ext_sample/     — minimal reference extension (not linked in main.cpp)
```

### Extension System

Static plugin architecture (no dynamic loading, WASM compatible). Core modules define interfaces; extensions inherit and register via `System`.

| Interface | Module | Purpose |
|-----------|--------|---------|
| `IExtension` | core_system | Entry point: `Register(System&)` — registers components, simulators, renderers |
| `ISimulator` | core_simulate | Per-frame simulation: `Initialize()`, `Update()`, `OnDatabaseChanged()` |
| `IDynamicsTerm` | core_simulate | Newton physics term: `DeclareSparsity()`, `Assemble()` |
| `IDynamicsTermProvider` | core_simulate | Newton term factory: `CreateTerm()`, `DeclareTopology()`, `QueryTopology()` |
| `IProjectiveTerm` | core_simulate | PD constraint term: `AssembleLHS()`, `ProjectRHS()` (fused local projection + RHS) |
| `IProjectiveTermProvider` | core_simulate | PD term factory: `CreateTerm()`, `DeclareTopology()`, `QueryTopology()` |
| `IObjectRenderer` | core_render | Rendering: `Render(RenderEngine&, WGPURenderPassEncoder)` |

### Third-Party Dependencies (`third_party/`, git submodules)

- **Dawn** — WebGPU + GLFW windowing (native only) | **GLM** — math (all platforms)

### Cross-Platform Pattern

Abstract interface (`IWindow`) + factory method (`Create()`) + separate `_native`/`_wasm` files. Platform selection via `#ifdef __EMSCRIPTEN__`.

### Namespaces

`mps` (primitives from types.h) | `mps::util` (math types, logger) | `mps::platform` (core_platform) | `mps::gpu` (core_gpu) | `mps::render` (core_render) | `mps::database` (core_database) | `mps::simulate` (core_simulate) | `mps::system` (core_system) | `ext_newton` / `ext_pd` / `ext_dynamics` / `ext_mesh` / `ext_sample` (extensions — not under `mps`)

## Key Coding Conventions

> Detailed standards: `.claude/agents/module-management.md`

- **C++20** strictly enforced
- **Naming**: `PascalCase` classes/methods, `snake_case_` private members (trailing `_`), `IPrefix` interfaces
- **Types**: Primitives (`uint32`, `float32`, etc.) are in `mps` namespace — no `util::` prefix needed. Math types (`util::vec3`, `util::mat4`) remain in `mps::util`. Never use raw `int`/`float`/`size_t`.
- **Headers**: `#pragma once`; project headers first, then STL
- **Memory**: `std::unique_ptr` ownership, raw pointers only for non-owning refs
- **Logging**: `LogInfo(...)`, `LogError(...)` from `core_util/logger.h`
- **Style**: 4 spaces (no tabs), no indentation inside namespaces
- **`using namespace`**: OK in `.cpp`, never in headers

## Commit Messages

> Detailed guide: `/commit-guide`

```
<type>(<scope>): <short description>
```

Types: `feat` | `fix` | `refactor` | `docs` | `style` | `test` | `chore`
Scope (optional): `core_util` | `core_platform` | `core_gpu` | `core_database` | `core_render` | `core_simulate` | `core_system` | `ext_newton` | `ext_pd` | `ext_dynamics` | `ext_mesh` | `ext_sample` | *(omit for project-wide)*

## Module Reference Documentation

**CRITICAL — Read docs FIRST, not source files:**

1. **ALWAYS** start by reading `.claude/docs/<module>.md` for any module you need to understand. These files contain complete file trees, type definitions, API surfaces, and shader references.
2. **DO NOT** read source files (`.h`, `.cpp`) to understand a module's structure or API. The docs already have this information and reading source files wastes tokens.
3. **ONLY** read actual source files when you need to see implementation details for a specific function or need to edit the file.

Doc files: `.claude/docs/core_util.md`, `core_platform.md`, `core_gpu.md`, `core_database.md`, `core_render.md`, `core_simulate.md`, `core_system.md`, `ext_newton.md`, `ext_pd.md`, `ext_dynamics.md`, `ext_mesh.md`, `ext_sample.md`

Agent files (`.claude/agents/*.md`) contain task instructions and workflow guidelines. Use `/sync-docs` to keep documentation synchronized with the codebase.

## Agent System

Native custom subagents in `.claude/agents/*.md` (YAML frontmatter). Auto-delegated by `description` field. Agents contain task guidelines and conventions; module structure/API reference is in `.claude/docs/*.md`.

| Agent | Scope | Modules |
|-------|-------|---------|
| agent-manager | Agent system, CLAUDE.md | — |
| dev-environment | CMake, Git, dependencies | — |
| build-debug | Build, run, debug, fix runtime errors | — (cross-module) |
| module-management | Architecture, coding standards, types | core_util, core_platform |
| gpu | WebGPU / Dawn, builders, surface, shaders | core_gpu |
| database | Host ECS, transactions, undo/redo | core_database |
| render | Rendering pipeline, camera, post-processing | core_render |
| simulate | Device DB, CG solver, dynamics term interfaces | core_simulate |
| system | System controller, ECS orchestration | core_system |

### Skills (Slash Commands)

| Skill | Purpose |
|-------|---------|
| `/new-module [name]` | Create a new module with cross-platform structure and CMake |
| `/add-dep [lib] [url]` | Add a third-party dependency (git submodule + CMake) |
| `/type-ref` | Type system & math reference (primitives, vectors, matrices) |
| `/review` | Code review checklist (naming, style, memory, const-correctness) |
| `/style` | C++20 code style & formatting guide (file layout, const-correctness, initialization) |
| `/commit-guide` | Commit message style guide (type selection, scope, format rules) |
| `/sync-docs` | Scan codebase and update all .md files to reflect current project state |
| `/verify` | Build + run verification for both native (Windows) and WASM platforms |
