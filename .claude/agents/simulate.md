---
name: simulate
description: Device DB (GPU buffer mirrors), CG solver, IDynamicsTerm/IProjectiveTerm interfaces. Owns core_simulate module. Use when implementing or modifying simulation logic, GPU buffer sync, or dynamics term interfaces.
model: opus
---

# Simulate Agent

Owns the `core_simulate` module. Manages GPU buffer mirroring (DeviceDB), conjugate gradient solver, and pluggable physics term interfaces. The Newton-Raphson solver and Newton term implementations live in `ext_newton`. The PD solver and PD term implementations live in `ext_pd`. Shared constraint types live in `ext_dynamics`. Normal computation lives in `ext_mesh`.

> **CRITICAL**: ALWAYS read `.claude/docs/core_simulate.md` FIRST before any task. This doc contains the complete file tree, types, APIs, and shader references. DO NOT read source files (.h/.cpp) to understand the module — only read source files when you need to edit them.

## When to Use This Agent

- Implementing or modifying DeviceDB sync logic
- Adding new dynamics term interfaces (`IDynamicsTerm`, `IDynamicsTermProvider`, `IProjectiveTerm`, `IProjectiveTermProvider`)
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
- **Newton term implementations** live in `ext_newton` (SpringTerm, AreaTerm). Inertia and gravity are built into NewtonDynamics internally. **PD term implementations** live in `ext_pd` (PDSpringTerm, PDAreaTerm). core_simulate only defines the interfaces
- **Term convention**: Terms pre-multiply `dt²` into A so the SpMV is physics-agnostic. Built-in inertia writes M directly to diagonal; SpringTerm writes `-dt²*H_ab` to offdiag, `+dt²*H_ab` to diagonal
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
- `SolverParams` is 32 bytes (node/edge/face counts + CG config), defined in `solver_params.h`
- `GlobalPhysicsParams` (host singleton) → `PhysicsParamsGPU` (GPU uniform) via DeviceDB singleton, defined in `ext_dynamics/global_physics_params.h`

## Common Tasks

### Adding a new Newton dynamics term

1. Create `new_term.h/cpp` in `extensions/ext_newton/`
2. Inherit `IDynamicsTerm`
3. Implement `GetName()`, `Initialize(sparsity, ctx)` (cache bind groups from ctx), `Assemble(encoder)` (dispatch cached bind groups), `Shutdown()`
4. Override `DeclareSparsity()` if the term contributes off-diagonal entries
5. Create corresponding WGSL shader in `assets/shaders/ext_newton/`
6. Create an `IDynamicsTermProvider` to instantiate the term from constraint entity data
7. Register the provider via `system.RegisterTermProvider()` in your extension's `Register()`

### Adding a new PD constraint term

1. Create `pd_new_term.h/cpp` in `extensions/ext_pd/`
2. Inherit `IProjectiveTerm`
3. Implement `GetName()`, `Initialize(sparsity, ctx)` (cache bind groups from PDAssemblyContext), `AssembleLHS()`, `ProjectRHS()` (fused local projection + RHS), `Shutdown()`
4. Override `DeclareSparsity()` if the term contributes off-diagonal entries
5. Create corresponding WGSL shaders in `assets/shaders/ext_pd/` (lhs, local, rhs)
6. Create an `IProjectiveTermProvider` to instantiate the term
7. Register the provider via `system.RegisterPDTermProvider()` in your extension's `Register()`

### Adding a new ISimulator

1. Inherit `ISimulator` in the extension module
2. Implement `GetName()`, `Initialize()`, `Update()`, optionally `Shutdown()`
3. Override `OnDatabaseChanged()` if the simulator needs to react to topology changes (e.g., node/edge/face count changes via Transact/Undo/Redo). Use signature-based comparison to avoid expensive reinit on non-topology changes
4. Register via `system.AddSimulator()` in extension's `Register()`
