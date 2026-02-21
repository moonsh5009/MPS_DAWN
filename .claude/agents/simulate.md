---
name: simulate
description: Device DB (GPU buffer mirrors), CG solver, IDynamicsTerm interface. Owns core_simulate module. Use when implementing or modifying simulation logic, GPU buffer sync, or dynamics term interfaces.
model: opus
---

# Simulate Agent

Owns the `core_simulate` module. Manages GPU buffer mirroring (DeviceDB), conjugate gradient solver, and pluggable physics term interfaces. The actual Newton-Raphson solver and all term implementations live in extensions (`ext_newton`, `ext_dynamics`). Normal computation lives in `ext_mesh`.

> **CRITICAL**: ALWAYS read `.claude/docs/core_simulate.md` FIRST before any task. This doc contains the complete file tree, types, APIs, and shader references. DO NOT read source files (.h/.cpp) to understand the module — only read source files when you need to edit them.

## When to Use This Agent

- Implementing or modifying DeviceDB sync logic
- Adding new dynamics term interfaces (`IDynamicsTerm`, `IDynamicsTermProvider`)
- Working with the CG solver
- Modifying CSR sparsity pattern construction
- Working with ISimulator interface

## Task Guidelines

### Sync Strategy

- DeviceDB does full re-upload of dirty dense arrays via `GPUBuffer<T>::WriteData()`
- Buffer usage: base `Storage | CopySrc | CopyDst`, plus extra flags from `Register()`
- Lazy buffer creation: buffer created/resized on first `SyncFromHost()` call
- Dirty tracking: uses `Database::GetDirtyTypeIds()` + `ClearAllDirty()` after sync
- `ForceSync()` re-uploads all registered types ignoring dirty flags (used by simulation reset)

### Dynamics Design Rules

- **NewtonDynamics** lives in `ext_newton` (NOT in core_simulate) — it's a helper class that computes `dv_total`. The caller (`ext_newton::NewtonSystemSimulator`) applies velocity/position updates
- **Term implementations** live in `ext_dynamics` (InertialTerm, GravityTerm, SpringTerm, AreaTerm). core_simulate only defines the interfaces
- **Term convention**: Terms pre-multiply `dt²` into A so the SpMV is physics-agnostic. InertialTerm writes M directly to diagonal; SpringTerm writes `-dt²*H_ab` to offdiag, `+dt²*H_ab` to diagonal
- **MPCG filter**: Zero residual/direction for pinned nodes (`inv_mass == 0`) in CG loop
- **ISpMVOperator**: Implementors cache bind groups via `PrepareSolve()` and dispatch via `Apply()`. This avoids re-creating bind groups every CG iteration
- **Bind group caching**: All bind groups are created once at Initialize time and reused across frames. `AssemblyContext` carries buffer handles for terms to cache their bind groups. `CGSolver::CacheBindGroups()` caches all CG bind groups. Topology changes trigger full reinit via `Shutdown()` + `Initialize()`

### Data Flow

- DeviceDB does NOT own the Database — takes a reference
- core_render reads GPU buffer handles but has NO dependency on core_simulate
- core_system bridges: calls `Sync()` after every Transact/Undo/Redo, passes buffer handles to render

### Namespace and Types

- Namespace: `mps::simulate`
- Use `mps::uint32`, `mps::float32` from `core_util/types.h`
- `DynamicsParams` is 48 bytes, layout-compatible with WGSL `SolverParams`

## Common Tasks

### Adding a new dynamics term

1. Create `new_term.h/cpp` in an extension (e.g., `extensions/ext_dynamics/`)
2. Inherit `IDynamicsTerm`
3. Implement `GetName()`, `Initialize(sparsity, ctx)` (cache bind groups from ctx), `Assemble(encoder)` (dispatch cached bind groups), `Shutdown()`
4. Override `DeclareSparsity()` if the term contributes off-diagonal entries
5. Create corresponding WGSL shader in the extension's shader dir (e.g., `assets/shaders/ext_dynamics/`)
6. Create an `IDynamicsTermProvider` to instantiate the term from constraint entity data
7. Register the provider via `system.RegisterTermProvider()` in your extension's `Register()`

### Adding a new ISimulator

1. Inherit `ISimulator` in the extension module
2. Implement `GetName()`, `Initialize()`, `Update(float32 dt)`, optionally `Shutdown()`
3. Override `OnDatabaseChanged()` if the simulator needs to react to topology changes (e.g., node/edge/face count changes via Transact/Undo/Redo). Use signature-based comparison to avoid expensive reinit on non-topology changes
4. Register via `system.AddSimulator()` in extension's `Register()`
