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
├── device_db.h / device_db.cpp # DeviceDB (GPU mirrors of host ECS data)
├── simulator.h                 # ISimulator interface (for extensions)
├── dynamics_term.h / .cpp      # IDynamicsTerm, AssemblyContext, SparsityBuilder
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
| `DeviceDB` | `device_db.h` | Registers component types, syncs dirty data to GPU |
| `ISimulator` | `simulator.h` | Extension interface: `Initialize()`, `Update(dt)`, `OnDatabaseChanged()` |
| `IDynamicsTerm` | `dynamics_term.h` | Pluggable physics contribution: `DeclareSparsity()`, `Initialize(sparsity, ctx)`, `Assemble(encoder)` |
| `AssemblyContext` | `dynamics_term.h` | GPU buffer handles passed to terms during Initialize for bind group caching |
| `SparsityBuilder` | `dynamics_term.h` | Builds CSR sparsity pattern from declared edges |
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

void Sync();       // Upload dirty types, then ClearAllDirty()
void ForceSync();  // Re-upload all registered types (ignores dirty flags)

template<database::Component T>
WGPUBuffer GetBufferHandle() const;

// Array queries
template<database::Component T>
uint32 GetArrayTotalCount() const;  // Total elements across all entities

IDeviceBufferEntry* GetEntryById(database::ComponentTypeId id) const;
bool IsRegistered(database::ComponentTypeId id) const;
```

### ISimulator

```cpp
virtual const std::string& GetName() const = 0;
virtual void Initialize() {}
virtual void Update(float32 dt) = 0;
virtual void Shutdown() {}
virtual void OnDatabaseChanged() {}  // Called after every Transact/Undo/Redo
```

### IDynamicsTerm

```cpp
virtual const std::string& GetName() const = 0;
virtual void DeclareSparsity(SparsityBuilder& builder) {}
virtual void Initialize(const SparsityBuilder& sparsity, const AssemblyContext& ctx) = 0;
virtual void Assemble(WGPUCommandEncoder encoder) = 0;
virtual void Shutdown() = 0;
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
void CacheBindGroups(WGPUBuffer params_buffer, uint64 params_size,
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
    WGPUBuffer position_buffer, velocity_buffer, mass_buffer;
    WGPUBuffer force_buffer, diag_buffer, csr_values_buffer;
    WGPUBuffer params_buffer, dv_total_buffer;
    uint32 node_count, edge_count, workgroup_size;
    uint64 params_size;
};
```

## Dynamics Architecture

```
IDynamicsTermProvider[] (registered with System)
├── GravityTermProvider    (ext_newton)    → creates GravityTerm
├── SpringTermProvider     (ext_dynamics)  → creates SpringTerm
└── AreaTermProvider       (ext_dynamics)  → creates AreaTerm

NewtonDynamics (ext_newton, instantiated by NewtonSystemSimulator)
├── SparsityBuilder (CSR pattern from edge declarations)
├── CGSolver + ISpMVOperator (linear solve)
└── IDynamicsTerm[] (pluggable physics, discovered from entity refs)
    ├── InertialTerm (ext_dynamics) — A_diag += M*I (always added)
    ├── GravityTerm  (ext_dynamics) — force += m*g
    ├── SpringTerm   (ext_dynamics) — A += -dt²*H, force += f_spring
    └── AreaTerm     (ext_dynamics) — force += f_area, A_diag += dt²*H_diag
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

Header includes (`assets/shaders/core_simulate/header/`): `atomic_float.wgsl` (CAS-based float atomics), `solver_params.wgsl` (SolverParams struct), `sim_mass.wgsl` (SimMass struct: `{mass, inv_mass}`).

Newton solver shaders are in `assets/shaders/ext_newton/`. Dynamics term shaders are in `assets/shaders/ext_dynamics/`. Normal computation shaders are in `assets/shaders/ext_mesh/`.
