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
emcmake cmake -B build-wasm && cmake --build build-wasm
# Output: build-wasm/bin/wasm/mps_dawn.html
```

No test framework yet; testing is manual (build and run).

## Architecture

C++20 WebGPU graphics engine using Dawn (native) and Emscripten (WASM). CMake static library modules linked via `mps::module_name` aliases.

### Module Layers (dependencies flow downward)

```
src/main.cpp (executable: mps_dawn)
  ├── core_system     (mps::core_system)   — controller: orchestrates database + simulate + render + extensions
  │     ├── core_database  (mps::core_database)  — host ECS (entities, components, transactions, undo/redo)
  │     ├── core_simulate  (mps::core_simulate)  — device DB (GPU buffer mirrors) + ISimulator interface
  │     └── core_render    (mps::core_render)    — rendering pipeline (camera, passes) + IObjectRenderer interface
  ├── core_gpu       (mps::core_gpu)       — WebGPU abstraction (device, buffers, shaders, textures, builders, surface)
  ├── core_platform  (mps::core_platform)  — window, input
  ├── core_util      (mps::core_util)      — types, logger, timer, math
  └── ext_sample     (mps::ext_sample)     — sample extension (links core_system)

extensions/ext_sample/   — static library plugin, registers components + simulator + renderer
```

### Extension System

Static plugin architecture (no dynamic loading, WASM compatible). Core modules define interfaces; extensions inherit and register via `System`.

| Interface | Module | Purpose |
|-----------|--------|---------|
| `IExtension` | core_system | Entry point: `Register(System&)` — registers components, simulators, renderers |
| `ISimulator` | core_simulate | Per-frame simulation: `Update(Database&, dt)`, wrapped in transaction |
| `IObjectRenderer` | core_render | Rendering: `Render(RenderEngine&, WGPURenderPassEncoder)` |

### Third-Party Dependencies (`third_party/`, git submodules)

- **Dawn** — WebGPU + GLFW windowing (native only) | **GLM** — math (all platforms)

### Cross-Platform Pattern

Abstract interface (`IWindow`) + factory method (`Create()`) + separate `_native`/`_wasm` files. Platform selection via `#ifdef __EMSCRIPTEN__`.

### Namespaces

`mps` (primitives from types.h) | `mps::util` (math types, logger) | `mps::platform` (core_platform) | `mps::gpu` (core_gpu) | `mps::render` (core_render) | `mps::database` (core_database) | `mps::simulate` (core_simulate) | `mps::system` (core_system) | `ext_sample` (sample extension — not under `mps`)

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
Scope (optional): `core_util` | `core_platform` | `core_gpu` | `core_database` | `core_render` | `core_simulate` | `core_system` | `ext_sample` | *(omit for project-wide)*

## Agent System

Native custom subagents in `.claude/agents/*.md` (YAML frontmatter). Auto-delegated by `description` field.

| Agent | Scope | Modules |
|-------|-------|---------|
| agent-manager | Agent system, CLAUDE.md | — |
| dev-environment | CMake, Git, dependencies | — |
| module-management | Architecture, coding standards, types | core_util, core_platform |
| gpu | WebGPU / Dawn, builders, surface, shaders | core_gpu |
| database | Host ECS, transactions, undo/redo | core_database |
| render | Rendering pipeline, camera, post-processing | core_render |
| simulate | Device DB, GPU buffer sync | core_simulate |
| system | System controller, ECS orchestration | core_system |

All agents use `memory: project` → `.claude/agent-memory/<name>/MEMORY.md` (first 200 lines auto-injected).

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
