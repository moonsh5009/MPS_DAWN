# core_simulate

> GPU buffer mirroring (DeviceDB), CG solver, and pluggable physics term interfaces.

## Module Structure

```
src/core_simulate/
├── CMakeLists.txt              # STATIC library → mps::core_simulate (depends: core_util, core_gpu, core_database)
├── sim_components.h            # SimPosition, SimVelocity, SimMass (generic particle components)
├── dynamics_term_provider.h    # IDynamicsTermProvider interface (term factory from entity data)
├── device_buffer_entry.h       # IDeviceBufferEntry, DeviceBufferEntry<T> (type-erased GPU buffer wrapper)
├── device_array_buffer.h       # IDeviceArrayEntry, DeviceArrayBuffer<T> (concatenated per-entity array GPU mirror)
├── device_db.h / device_db.cpp # DeviceDB (GPU mirrors of host ECS data + singleton uniforms)
├── simulator.h                 # ISimulator interface (for extensions)
├── simulate_config.h           # kEnableSimulationProfiling (compile-time toggle) + WaitForGPU() (GPU sync)
├── solver_params.h             # SolverParams (per-solver GPU uniform: topology counts + CG config)
├── dynamics_term.h / .cpp      # IDynamicsTerm, AssemblyContext, SparsityBuilder
├── projective_term.h           # IProjectiveTerm, PDAssemblyContext (PD constraint interface)
├── projective_term_provider.h  # IProjectiveTermProvider (PD term factory)
└── cg_solver.h / .cpp          # CGSolver + ISpMVOperator (conjugate gradient solver)
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `SimPosition` | `sim_components.h` | `{x, y, z, w}` — 16 bytes, generic particle position |
| `SimVelocity` | `sim_components.h` | `{vx, vy, vz, w}` — 16 bytes, generic particle velocity |
| `SimMass` | `sim_components.h` | `{mass, inv_mass}` — 8 bytes, `inv_mass=0` means pinned |
| `IDynamicsTermProvider` | `dynamics_term_provider.h` | Term factory interface: `HasConfig()`, `CreateTerm()`, `DeclareTopology()`, `QueryTopology()` |
| `IDeviceBufferEntry` | `device_buffer_entry.h` | Type-erased base for GPU buffer entries |
| `DeviceBufferEntry<T>` | `device_buffer_entry.h` | Owns `gpu::GPUBuffer<T>`, syncs from `IComponentStorage` |
| `IDeviceArrayEntry` | `device_array_buffer.h` | Type-erased base for concatenated array GPU buffer entries |
| `DeviceArrayBuffer<T>` | `device_array_buffer.h` | Concatenates per-entity `ArrayStorage<T>` into a single GPU buffer |
| `ArrayRegion` | `device_array_buffer.h` | Describes an entity's offset/count in the concatenated buffer |
| `ISingletonBufferEntry` | `device_db.h` | Type-erased base for singleton uniform buffer entries |
| `SingletonBufferEntry<HostT, GpuT>` | `device_db.h` | Transforms host singleton → GPU uniform, uploads on change |
| `DeviceDB` | `device_db.h` | Registers component types + singletons, syncs dirty data to GPU |
| `ISimulator` | `simulator.h` | Extension interface: `Initialize()`, `Update()`, `OnDatabaseChanged()` |
| `kEnableSimulationProfiling` | `simulate_config.h` | `inline constexpr bool` — compile-time toggle for GPU-synced profiling in simulator Update() |
| `WaitForGPU()` | `simulate_config.h` | Inline function — blocks until all submitted GPU work completes (QueueOnSubmittedWorkDone + WaitAny) |
| `IDynamicsTerm` | `dynamics_term.h` | Newton physics contribution: `DeclareSparsity()`, `Initialize(sparsity, ctx)`, `Assemble(encoder)` |
| `AssemblyContext` | `dynamics_term.h` | GPU buffer handles passed to Newton terms during Initialize for bind group caching |
| `SparsityBuilder` | `dynamics_term.h` | Builds CSR sparsity pattern from declared edges |
| `IProjectiveTerm` | `projective_term.h` | PD constraint term: `AssembleLHS()`, `ProjectRHS()` (fused local projection + RHS) |
| `PDAssemblyContext` | `projective_term.h` | GPU buffer handles passed to PD terms during Initialize for bind group caching |
| `IProjectiveTermProvider` | `projective_term_provider.h` | PD term factory: `HasConfig()`, `CreateTerm()`, `DeclareTopology()`, `QueryTopology()` |
| `SolverParams` | `solver_params.h` | Per-solver GPU uniform: `alignas(16) {node_count, edge_count, face_count, cg_max_iter, cg_tolerance}` — 32 bytes |
| `CGSolver` | `cg_solver.h` | Generic GPU conjugate gradient solver (MPCG for pinned nodes) |
| `ISpMVOperator` | `cg_solver.h` | Interface for sparse matrix-vector product: `PrepareSolve()` + `Apply()` |

## API

### SimPosition / SimVelocity / SimMass

```cpp
struct SimPosition { float32 x, y, z, w; };   // maps to vec4<f32>
struct SimVelocity { float32 vx, vy, vz, w; };
struct SimMass     { float32 mass, inv_mass; };  // inv_mass=0 → pinned (8 bytes)
```

### IDynamicsTermProvider

```cpp
virtual std::string_view GetTermName() const = 0;
virtual bool HasConfig(const database::Database& db, database::Entity entity) const = 0;
virtual std::unique_ptr<IDynamicsTerm> CreateTerm(
    const database::Database& db, database::Entity entity, uint32 node_count) = 0;
virtual void DeclareTopology(uint32& out_edge_count, uint32& out_face_count);
// Lightweight topology query (no GPU allocation). Override to check array sizes from DB.
virtual void QueryTopology(const database::Database& db, database::Entity entity,
                           uint32& out_edge_count, uint32& out_face_count) const;
```

### DeviceDB

```cpp
explicit DeviceDB(database::Database& host_db);

// Register component types for GPU mirroring (sparse-set data)
template<database::Component T>
void Register(gpu::BufferUsage extra_usage = gpu::BufferUsage::None,
              const std::string& label = "");

// Register array types for GPU mirroring (concatenated per-entity arrays)
template<database::Component T>
void RegisterArray(gpu::BufferUsage extra_usage = gpu::BufferUsage::None,
                   const std::string& label = "");

// Register indexed array type (topology arrays with per-entity offset transform)
template<database::Component T, database::Component RefT>
void RegisterIndexedArray(gpu::BufferUsage extra_usage, const std::string& label,
                          IndexOffsetFn<T> offset_fn);

void Sync();       // Upload dirty types, then ClearAllDirty()
void ForceSync();  // Re-upload all registered types (ignores dirty flags)

template<database::Component T>
WGPUBuffer GetBufferHandle() const;

// Array queries
template<database::Component T>
uint32 GetArrayTotalCount() const;  // Total elements across all entities

IDeviceBufferEntry* GetEntryById(database::ComponentTypeId id) const;
IDeviceArrayEntry* GetArrayEntryById(database::ComponentTypeId id) const;  // checks arrays + indexed

// Singleton uniform buffers (host → GPU transform)
template<Component HostT, typename GpuT>
void RegisterSingleton(std::function<GpuT(const HostT&)> transform,
                       const std::string& label = "");

template<Component HostT>
WGPUBuffer GetSingletonBuffer() const;

bool IsRegistered(database::ComponentTypeId id) const;
```

### ISimulator

```cpp
virtual const std::string& GetName() const = 0;
virtual void Initialize() {}
virtual void Update() = 0;
virtual void Shutdown() {}
virtual void OnDatabaseChanged() {}  // Called after every Transact/Undo/Redo
```

### Simulation Profiling (simulate_config.h)

```cpp
// Compile-time toggle — only included by simulator .cpp files.
// Changing this value only rebuilds those .cpp files (not all ISimulator dependents).
inline constexpr bool kEnableSimulationProfiling = true;

// Block until all previously submitted GPU work completes.
// Uses wgpuQueueOnSubmittedWorkDone + WaitAny (native) or ProcessEvents (WASM).
inline void WaitForGPU();
```

Usage in `ISimulator::Update()` implementations:

```cpp
if constexpr (kEnableSimulationProfiling) {
    WaitForGPU();           // drain previous GPU work
    profile_timer.Start();
}
// ... command recording + submit ...
if constexpr (kEnableSimulationProfiling) {
    WaitForGPU();           // wait for this frame's GPU work
    profile_timer.Stop();
    LogInfo("[Profile] ", kName, "::Update: ", profile_timer.GetElapsedMilliseconds(), " ms");
}
```

### IDynamicsTerm

```cpp
virtual const std::string& GetName() const = 0;
virtual void DeclareSparsity(SparsityBuilder& builder) {}
virtual void Initialize(const SparsityBuilder& sparsity, const AssemblyContext& ctx) = 0;
virtual void Assemble(WGPUCommandEncoder encoder) = 0;
virtual void Shutdown() = 0;
```

### IProjectiveTerm

```cpp
virtual const std::string& GetName() const = 0;
virtual void DeclareSparsity(SparsityBuilder& builder) {}
virtual void Initialize(const SparsityBuilder& sparsity, const PDAssemblyContext& ctx) = 0;
virtual void AssembleLHS(WGPUCommandEncoder encoder) = 0;   // constant LHS (w * S^T * S)
virtual void ProjectRHS(WGPUCommandEncoder encoder) = 0;    // fused local projection + RHS (w * S^T * p)
virtual void Shutdown() = 0;
```

### PDAssemblyContext

Passed to `IProjectiveTerm::Initialize()` for bind group caching.

```cpp
struct PDAssemblyContext {
    WGPUBuffer physics_buffer;     // global physics params uniform (binding 0)
    WGPUBuffer q_buffer;           // current iterate q (read)
    WGPUBuffer s_buffer;           // predicted positions s (read)
    WGPUBuffer mass_buffer;        // mass data (read)
    WGPUBuffer rhs_buffer;         // RHS accumulation (atomic u32, read_write)
    WGPUBuffer diag_buffer;        // LHS diagonal 3x3 blocks (atomic u32, read_write)
    WGPUBuffer csr_values_buffer;  // LHS off-diagonal CSR 3x3 blocks (read_write)
    WGPUBuffer params_buffer;      // solver params uniform (binding 1)
    uint32 node_count, edge_count, workgroup_size;
    uint64 physics_size;           // size of physics buffer in bytes
    uint64 params_size;            // size of solver params buffer in bytes
};
```

### IProjectiveTermProvider

```cpp
virtual std::string_view GetTermName() const = 0;
virtual bool HasConfig(const database::Database& db, database::Entity entity) const = 0;
virtual std::unique_ptr<IProjectiveTerm> CreateTerm(
    const database::Database& db, database::Entity entity, uint32 node_count) = 0;
virtual void DeclareTopology(uint32& out_edge_count, uint32& out_face_count);
virtual void QueryTopology(const database::Database& db, database::Entity entity,
                           uint32& out_edge_count, uint32& out_face_count) const;
```

### SparsityBuilder

```cpp
explicit SparsityBuilder(uint32 node_count);
void AddEdge(uint32 node_a, uint32 node_b);
void Build();
const std::vector<uint32>& GetRowPtr() const;
const std::vector<uint32>& GetColIdx() const;
uint32 GetNNZ() const;
uint32 GetNodeCount() const;
uint32 GetCSRIndex(uint32 row, uint32 col) const;
```

### CGSolver

```cpp
void Initialize(uint32 node_count, uint32 workgroup_size = 64);

WGPUBuffer GetRHSBuffer() const;
WGPUBuffer GetSolutionBuffer() const;
uint64 GetVectorSize() const;

// Cache all bind groups (call after Initialize, before Solve)
void CacheBindGroups(WGPUBuffer physics_buffer, uint64 physics_size,
                     WGPUBuffer params_buffer, uint64 params_size,
                     WGPUBuffer mass_buffer, uint64 mass_size,
                     ISpMVOperator& spmv);

// Dispatch CG iterations using cached bind groups
void Solve(WGPUCommandEncoder encoder, uint32 cg_iterations);

void Shutdown();
```

### ISpMVOperator

```cpp
virtual void PrepareSolve(WGPUBuffer p_buffer, uint64 p_size,
                          WGPUBuffer ap_buffer, uint64 ap_size) = 0;
virtual void Apply(WGPUCommandEncoder encoder, uint32 workgroup_count) = 0;
```

### AssemblyContext

Passed to `IDynamicsTerm::Initialize()` for bind group caching. Terms cache bind groups from these handles and reuse them in `Assemble()`.

```cpp
struct AssemblyContext {
    WGPUBuffer physics_buffer;      // global physics params uniform (binding 0)
    WGPUBuffer position_buffer, velocity_buffer, mass_buffer;
    WGPUBuffer force_buffer, diag_buffer, csr_values_buffer;
    WGPUBuffer params_buffer;       // solver params uniform (binding 1)
    WGPUBuffer dv_total_buffer;
    uint32 node_count, edge_count, workgroup_size;
    uint64 physics_size;            // size of physics buffer in bytes
    uint64 params_size;             // size of solver params buffer in bytes
};
```

## Dynamics Architecture

### Newton Solver (ext_newton)

```
IDynamicsTermProvider[] (registered with System)
├── SpringTermProvider     (ext_newton) → creates SpringTerm
└── AreaTermProvider       (ext_newton) → creates AreaTerm

NewtonDynamics (ext_newton, instantiated by NewtonSystemSimulator)
├── SparsityBuilder (CSR pattern from edge declarations)
├── CGSolver + ISpMVOperator (linear solve)
├── Built-in: inertia (A_diag += M*I) + gravity (force += m*g)
└── IDynamicsTerm[] (pluggable physics, discovered from entity refs)
    ├── SpringTerm   (ext_newton) — A += -dt²*H, force += f_spring
    └── AreaTerm     (ext_newton) — force += f_area, A_diag += dt²*H_diag
```

### PD Solver (ext_pd)

```
IProjectiveTermProvider[] (registered with System)
├── PDSpringTermProvider   (ext_pd) → creates PDSpringTerm
└── PDAreaTermProvider     (ext_pd) → creates PDAreaTerm

PDDynamics (ext_pd, instantiated by PDSystemSimulator)
├── SparsityBuilder (CSR pattern from edge declarations)
└── IProjectiveTerm[] (pluggable constraints, discovered from entity refs)
    ├── PDSpringTerm (ext_pd) — spring distance constraint
    └── PDAreaTerm   (ext_pd) — ARAP area constraint
```

## Shaders (`assets/shaders/core_simulate/`)

| Shader | Used by | Purpose |
|--------|---------|---------|
| `cg_init.wgsl` | CGSolver | Initialize CG vectors (r, p, x) |
| `cg_dot.wgsl` | CGSolver | Partial dot product reduction |
| `cg_dot_final.wgsl` | CGSolver | Final dot product reduction |
| `cg_compute_scalars.wgsl` | CGSolver | Compute alpha/beta scalars |
| `cg_update_xr.wgsl` | CGSolver | Update solution and residual |
| `cg_update_p.wgsl` | CGSolver | Update search direction |

Header includes (`assets/shaders/core_simulate/header/`): `physics_params.wgsl` (PhysicsParams struct: dt, gravity, damping, derived values), `solver_params.wgsl` (SolverParams struct: topology counts + CG config), `atomic_float.wgsl` (CAS-based float atomics), `sim_mass.wgsl` (SimMass struct: `{mass, inv_mass}`).

Newton solver + dynamics term shaders are in `assets/shaders/ext_newton/`. PD solver + projective term shaders are in `assets/shaders/ext_pd/`. Normal computation shaders are in `assets/shaders/ext_mesh/`.
