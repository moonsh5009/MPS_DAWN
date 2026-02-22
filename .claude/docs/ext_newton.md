# ext_newton

> Newton-Raphson dynamics extension — solver, all IDynamicsTerm implementations, and data-driven constraint term discovery.

## Module Structure

```
extensions/ext_newton/
├── CMakeLists.txt                   # STATIC library → mps::ext_newton (depends: mps::core_system, mps::ext_dynamics)
├── newton_dynamics.h / .cpp         # NewtonDynamics solver (Newton-Raphson + CSR + SpMV + built-in inertia/gravity)
├── newton_system_config.h           # NewtonSystemConfig component (solver params + constraint entity refs)
├── newton_extension.h / .cpp        # NewtonExtension (IExtension) — registers providers, simulator
├── newton_system_simulator.h / .cpp # NewtonSystemSimulator (ISimulator) — discovers terms, runs solver, integrates
├── gravity_constraint.h             # GravityConstraintData component (legacy, unused)
├── spring_term_provider.h / .cpp    # SpringTermProvider (IDynamicsTermProvider → SpringTerm)
├── spring_term.h / .cpp             # SpringTerm (IDynamicsTerm) — spring Hessian + force assembly
├── area_term_provider.h / .cpp      # AreaTermProvider (IDynamicsTermProvider → AreaTerm)
└── area_term.h / .cpp               # AreaTerm (IDynamicsTerm) — area preservation force + full Hessian
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `NewtonSystemConfig` | `newton_system_config.h` | Host-only config: solver params + constraint entity refs (64 bytes) |
| `NewtonExtension` | `newton_extension.h` | IExtension: registers term providers, simulator |
| `NewtonDynamics` | `newton_dynamics.h` | Newton-Raphson solver with pluggable terms, built-in inertia/gravity, internal SpMV |
| `NewtonSystemSimulator` | `newton_system_simulator.h` | ISimulator: discovers terms via System registry, runs NewtonDynamics, integrates velocity/position |
| `SpringTermProvider` | `spring_term_provider.h` | IDynamicsTermProvider: creates SpringTerm from ArrayStorage\<SpringEdge\> |
| `SpringParams` | `spring_term.h` | GPU uniform: `alignas(16) {stiffness}` — 16 bytes |
| `SpringTerm` | `spring_term.h` | IDynamicsTerm: spring Hessian (`-dt²*H`) + force assembly |
| `AreaTermProvider` | `area_term_provider.h` | IDynamicsTermProvider: creates AreaTerm from ArrayStorage\<AreaTriangle\> |
| `AreaParams` | `area_term.h` | GPU uniform: `alignas(16) {stiffness, shear_stiffness}` — 16 bytes |
| `AreaTerm` | `area_term.h` | IDynamicsTerm: area preservation force + full Hessian (diagonal + off-diagonal CSR) |

## API

### NewtonExtension

```cpp
explicit NewtonExtension(mps::system::System& system);
const std::string& GetName() const override;       // "ext_newton"
void Register(mps::system::System& system) override;
```

`Register()` does:
1. `RegisterTermProvider(SpringConstraintData → SpringTermProvider)`
2. `RegisterTermProvider(AreaConstraintData → AreaTermProvider)`
3. `AddSimulator(NewtonSystemSimulator)`

### NewtonSystemSimulator

```cpp
explicit NewtonSystemSimulator(mps::system::System& system);
~NewtonSystemSimulator() override;
const std::string& GetName() const override;       // "NewtonSystemSimulator"
void Initialize() override;
void Update() override;
void Shutdown() override;
void OnDatabaseChanged() override;  // Signature-based topology change detection + reinit
```

`Initialize()` flow:
1. Query Database for entities with `NewtonSystemConfig`
2. Read `constraint_entities[]` array from first config
3. For each constraint entity, call `System::FindAllTermProviders()` → `provider->CreateTerm()`
4. Get GPU buffer handles (pos, vel, mass) from DeviceDB; physics buffer from DeviceDB singleton
5. Call `NewtonDynamics::Initialize(node_count, edges, faces, physics_buf, physics_sz, pos, vel, mass)`
6. Create update_velocity and update_position compute pipelines
7. Cache velocity/position update bind groups (`bg_vel_`, `bg_pos_`)
8. Cache `TopologySignature` for change detection

`Update()` flow:
1. Create command encoder
2. Copy-in (scoped mode: global → local buffers)
3. `NewtonDynamics::Solve(encoder)` → computes dv_total (uses cached bind groups)
4. Dispatch cached `bg_vel_`: `v = (v + dv_total) * damping`
5. Dispatch cached `bg_pos_`: `pos = x_old + v * dt`
6. Copy-out (scoped mode: local → global buffers)

`OnDatabaseChanged()` flow:
1. Compute new `TopologySignature` (node_count, total_edges, total_faces, constraint_count)
2. If not initialized and nodes exist → try `Initialize()`
3. Compare with cached signature → skip if unchanged
4. If changed → `Shutdown()` + `Initialize()`

### NewtonSystemConfig (64 bytes, host-only)

```cpp
struct NewtonSystemConfig {
    static constexpr uint32 MAX_CONSTRAINTS = 8;
    uint32 newton_iterations  = 1;
    uint32 cg_max_iterations  = 30;
    float32 cg_tolerance      = 1e-6f;
    uint32 constraint_count   = 0;
    uint32 constraint_entities[MAX_CONSTRAINTS] = {};
    uint32 mesh_entity        = database::kInvalidEntity;  // kInvalidEntity = global, valid entity = scoped
    uint32 padding[3]         = {};
};
```

### SpringTermProvider

```cpp
std::string_view GetTermName() const override;                    // "SpringTermProvider"
bool HasConfig(const Database& db, Entity entity) const override; // checks SpringConstraintData
std::unique_ptr<IDynamicsTerm> CreateTerm(
    const Database& db, Entity entity, uint32 node_count) override;
void DeclareTopology(uint32& out_edge_count, uint32& out_face_count) override;
void QueryTopology(const Database& db, Entity entity,
                   uint32& out_edge_count, uint32& out_face_count) const override;
```

`CreateTerm()` reads `ArrayStorage<SpringEdge>` from the entity. Returns `nullptr` if no edges.

### SpringTerm

```cpp
SpringTerm(const std::vector<ext_dynamics::SpringEdge>& edges, mps::float32 stiffness);
const std::string& GetName() const override;
void DeclareSparsity(SparsityBuilder& builder) override;
void Initialize(const SparsityBuilder& sparsity, const AssemblyContext& ctx) override;
void Assemble(WGPUCommandEncoder encoder) override;
void Shutdown() override;
```

### AreaTermProvider

```cpp
std::string_view GetTermName() const override;                    // "AreaTermProvider"
bool HasConfig(const Database& db, Entity entity) const override; // checks AreaConstraintData
std::unique_ptr<IDynamicsTerm> CreateTerm(
    const Database& db, Entity entity, uint32 node_count) override;
void DeclareTopology(uint32& out_edge_count, uint32& out_face_count) override;
void QueryTopology(const Database& db, Entity entity,
                   uint32& out_edge_count, uint32& out_face_count) const override;
```

`CreateTerm()` reads `ArrayStorage<AreaTriangle>` and `AreaConstraintData::stiffness`. Returns `nullptr` if no triangles.

### AreaTerm

```cpp
AreaTerm(const std::vector<ext_dynamics::AreaTriangle>& triangles, float32 stiffness);
const std::string& GetName() const override;
void DeclareSparsity(SparsityBuilder& builder) override;
void Initialize(const SparsityBuilder& sparsity, const AssemblyContext& ctx) override;
void Assemble(WGPUCommandEncoder encoder) override;
void Shutdown() override;
```

### NewtonDynamics

```cpp
void AddTerm(std::unique_ptr<IDynamicsTerm> term);

// Configure solver iterations (call before Initialize or anytime)
void SetNewtonIterations(uint32 iterations);
void SetCGMaxIterations(uint32 iterations);

// Initialize with external buffer handles for bind group caching.
// physics_buffer: DeviceDB singleton uniform (binding 0).
void Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                WGPUBuffer physics_buffer, uint64 physics_size,
                WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                WGPUBuffer mass_buffer, uint32 workgroup_size = 64);

// Run Newton-Raphson solver (uses cached bind groups, no per-frame params)
void Solve(WGPUCommandEncoder encoder);

WGPUBuffer GetDVTotalBuffer() const;
WGPUBuffer GetXOldBuffer() const;
WGPUBuffer GetParamsBuffer() const;
uint64 GetParamsSize() const;
uint64 GetVec4BufferSize() const;
void Shutdown();
```

## Entity Model

```
Newton System Entity:
  └── NewtonSystemConfig { newton_iterations, cg_max_iterations, constraint_refs: [Entity 200] }

Constraint Entity (mesh entity with constraint data):
  └── Entity 200: SpringConstraintData + AreaConstraintData (from ext_dynamics)

Gravity is read from GlobalPhysicsParams DB singleton (ext_dynamics/global_physics_params.h).
```

## Shaders (`assets/shaders/ext_newton/`)

| Shader | Used by | Purpose |
|--------|---------|---------|
| `update_velocity.wgsl` | NewtonSystemSimulator | Apply velocity delta: `v = (v + dv_total) * damping` |
| `update_position.wgsl` | NewtonSystemSimulator | Integrate positions: `pos = x_old + v * dt` |
| `newton_init.wgsl` | NewtonDynamics | Initialize Newton iteration (save x_old, zero dv_total) |
| `newton_predict_pos.wgsl` | NewtonDynamics | Predict positions from velocities |
| `newton_accumulate_dv.wgsl` | NewtonDynamics | Accumulate CG solution into dv_total |
| `clear_forces.wgsl` | NewtonDynamics | Zero force buffer |
| `assemble_rhs.wgsl` | NewtonDynamics | Assemble RHS for CG solve |
| `cg_spmv.wgsl` | NewtonDynamics | Sparse matrix-vector product |
| `inertia_assemble.wgsl` | NewtonDynamics | Write mass to A diagonal: `A_ii += M_i * I3x3` |
| `accumulate_gravity.wgsl` | NewtonDynamics | Add gravity forces: `force[i] += mass_i * g` |
| `accumulate_springs.wgsl` | SpringTerm | Spring force + Hessian assembly (`-dt²*H` off-diagonal, `+dt²*H` diagonal) |
| `accumulate_area.wgsl` | AreaTerm | Area preservation force + full Hessian (FEM/SVD + ARAP shear) |

All shaders import `core_simulate/header/solver_params.wgsl` (root-relative). Shaders that use physics params also import `core_simulate/header/physics_params.wgsl`. CG solver shaders are in `assets/shaders/core_simulate/`.
