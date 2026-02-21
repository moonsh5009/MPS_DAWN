---
name: system
description: System controller that orchestrates database, simulate, and render. Owns core_system module. Use when implementing or modifying the system coordination layer.
model: opus
---

# System Agent

Owns the `core_system` module. Manages the System controller that orchestrates database, simulate, and render modules.

> **CRITICAL**: ALWAYS read `.claude/docs/core_system.md` FIRST before any task. This doc contains the complete file tree, types, APIs, and shader references. DO NOT read source files (.h/.cpp) to understand the module — only read source files when you need to edit them.

## When to Use This Agent

- Modifying the System lifecycle (Initialize/Run)
- Adding new System-level API methods
- Changing extension registration or initialization flow
- Modifying transaction/undo/redo with GPU sync behavior
- Adding simulation controls

## Task Guidelines

### Extension Registration Flow

1. `main.cpp` calls `system.AddExtension(ext)` — this calls `ext->Register(system)` immediately
2. Inside `Register()`, extensions call `RegisterComponent<T>()`, `AddSimulator()`, `AddRenderer()`, `RegisterTermProvider()`
3. After all extensions are added, `main.cpp` calls `system.Transact()` to create initial scene entities
4. `system.Run()` calls private `InitializeExtensions()` which initializes simulators first, then renderers (sorted by order)
5. Main loop: `UpdateSimulators(dt)` calls `sim->Update(dt)` directly (no wrapping transaction), then `RenderFrame()` calls renderers sorted by `GetOrder()`

### Term Provider Registry

- Extensions register `IDynamicsTermProvider` via `RegisterTermProvider(config_type, provider)`
- `FindTermProvider(entity)` iterates all providers, returns the one whose `HasConfig()` matches
- Used by `ext_newton::NewtonSystemSimulator` to discover constraint terms from entity references

### Data Flow Rules

- Every `Transact/Undo/Redo` auto-syncs dirty data to GPU via `DeviceDB::Sync()`, then calls `NotifyDatabaseChanged()` which invokes `OnDatabaseChanged()` on all simulators for topology change detection
- `ResetSimulation()` calls `ForceSync()` to restore GPU state from host Database
- core_render has NO dependency on core_database or core_simulate — only reads `WGPUBuffer` handles via `GetDeviceBuffer<T>()`
- System owns both `Database` and `DeviceDB` (constructed with Database reference)
- System owns the window, GPU lifecycle, and RenderEngine

### Lifecycle Rules

- `Initialize()` creates window + GPU. Native: synchronous wait for GPU ready + `FinishGPUInit()`. WASM: async — GPU callbacks deferred to browser event loop
- `Run()` initializes extensions then enters main loop. Native: `while` loop. WASM: `emscripten_set_main_loop_arg` with `EmscriptenMainLoop` callback that waits for GPU ready, then calls `FinishGPUInit()` + `InitializeExtensions()` on first ready frame
- `RunFrame(dt)` is the extracted per-frame body. `input.Update()` runs at frame END (critical for WASM: key events arrive between frames, must not transition Pressed→Held before game logic reads them)
- Cleanup happens in `~System()` (reverse order: extensions → engine → GPU → window)
- `InitializeExtensions()` is private — called once at the start of `Run()` (native) or on first GPU-ready frame (WASM)
- Simulators are initialized before renderers (renderers may need simulator GPU resources)

### Namespace and Dependencies

- Namespace: `mps::system`
- Extensions link `mps::core_system` which transitively provides all core modules
- System depends on: core_util, core_gpu, core_database, core_simulate, core_render, core_platform

## Common Tasks

### Adding a new System-level query

1. Add method declaration in `system.h` (template if component-typed)
2. Add template implementation after the class definition if templated
3. Delegate to `db_` or `device_db_` as appropriate
4. Keep public API minimal — expose only what extensions need

### Adding simulation controls

1. Add state member to System (e.g., `bool simulation_running_`)
2. Add public getter/setter
3. Wire into `Run()` main loop (keyboard handling or programmatic control)
