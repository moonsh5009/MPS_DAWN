# ext_newton

> Newton-Raphson dynamics extension — data-driven solver that discovers constraint terms from entity references.

## Module Structure

```
extensions/ext_newton/
├── CMakeLists.txt                   # STATIC library → mps::ext_newton (depends: mps::core_system, mps::ext_dynamics)
├── newton_dynamics.h / .cpp         # NewtonDynamics solver (Newton-Raphson + CSR + SpMV)
├── newton_system_config.h           # NewtonSystemConfig component (solver params + constraint entity refs)
├── gravity_constraint.h             # GravityConstraintData component (gravity vector)
├── gravity_term_provider.h / .cpp   # GravityTermProvider (IDynamicsTermProvider → GravityTerm)
├── newton_extension.h / .cpp        # NewtonExtension (IExtension) — registers components, provider, simulator
└── newton_system_simulator.h / .cpp # NewtonSystemSimulator (ISimulator) — discovers terms, runs solver, integrates
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `NewtonSystemConfig` | `newton_system_config.h` | Host-only config: solver params + constraint entity refs (64 bytes) |
| `GravityConstraintData` | `gravity_constraint.h` | Host-only gravity config: `{gx, gy, gz}` |
| `GravityTermProvider` | `gravity_term_provider.h` | IDynamicsTermProvider: creates GravityTerm from GravityConstraintData |
| `NewtonExtension` | `newton_extension.h` | IExtension: registers SimPosition/SimVelocity/SimMass, GravityTermProvider, NewtonSystemSimulator |
| `DynamicsParams` | `newton_dynamics.h` | 48-byte params struct (layout-compatible with WGSL SolverParams) |
| `NewtonDynamics` | `newton_dynamics.h` | Newton-Raphson solver with pluggable terms and internal SpMV |
| `NewtonSystemSimulator` | `newton_system_simulator.h` | ISimulator: discovers terms via System registry, runs NewtonDynamics, integrates velocity/position |

## API

### NewtonExtension

```cpp
explicit NewtonExtension(mps::system::System& system);
const std::string& GetName() const override;       // "ext_newton"
void Register(mps::system::System& system) override;
```

`Register()` does:
1. `RegisterComponent<SimPosition>(Storage | Vertex)` — GPU-synced
2. `RegisterComponent<SimVelocity>(Storage)` — GPU-synced
3. `RegisterComponent<SimMass>(Storage)` — GPU-synced
4. `RegisterComponent<NewtonSystemConfig>()` — host-only (no GPU sync)
5. `RegisterComponent<GravityConstraintData>()` — host-only
6. `RegisterTermProvider(GravityConstraintData → GravityTermProvider)`
7. `AddSimulator(NewtonSystemSimulator)`

### NewtonSystemSimulator

```cpp
explicit NewtonSystemSimulator(mps::system::System& system);
~NewtonSystemSimulator() override;
const std::string& GetName() const override;       // "NewtonSystemSimulator"
void Initialize() override;
void Update(mps::float32 dt) override;
void Shutdown() override;
void OnDatabaseChanged() override;  // Signature-based topology change detection + reinit
```

`Initialize()` flow:
1. Query Database for entities with `NewtonSystemConfig`
2. Read `constraint_entities[]` array from first config
3. Always add `InertialTerm` (fundamental to Newton method)
4. For each constraint entity, call `System::FindTermProvider()` → `provider->CreateTerm()`
5. Get GPU buffer handles (pos, vel, mass) from DeviceDB
6. Call `NewtonDynamics::Initialize(node_count, total_edges, total_faces, pos, vel, mass)`
7. Create update_velocity and update_position compute pipelines
8. Cache velocity/position update bind groups (`bg_vel_`, `bg_pos_`)
9. Cache `TopologySignature` for change detection

`Update(dt)` flow:
1. Create command encoder
2. `NewtonDynamics::Solve(encoder, dt, ...)` → computes dv_total (uses cached bind groups)
3. Dispatch cached `bg_vel_`: `v = (v + dv_total) * damping`
4. Dispatch cached `bg_pos_`: `pos = x_old + v * dt`

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
    float32 damping           = 0.999f;
    float32 cg_tolerance      = 1e-6f;
    uint32 constraint_count   = 0;
    uint32 constraint_entities[MAX_CONSTRAINTS] = {};
    uint32 padding[3]         = {};
};
```

### GravityConstraintData (host-only)

```cpp
struct GravityConstraintData {
    float32 gx = 0.0f;
    float32 gy = -9.81f;
    float32 gz = 0.0f;
};
```

### GravityTermProvider

```cpp
std::string_view GetTermName() const override;                    // "GravityTerm"
bool HasConfig(const Database& db, Entity entity) const override; // checks GravityConstraintData
std::unique_ptr<IDynamicsTerm> CreateTerm(
    const Database& db, Entity entity, uint32 node_count) override;
```

## Entity Model

```
Newton System Entity (e.g., Entity 100):
  └── NewtonSystemConfig { iterations, damping, constraint_refs: [Entity 200, Entity 201] }

Constraint Entities:
  ├── Entity 200: GravityConstraintData { gx, gy, gz }
  └── Entity 201: SpringConstraintData  { } (from ext_dynamics)

Particle Entities (Entity 0..N-1):
  ├── SimPosition { x, y, z, w }
  ├── SimVelocity { vx, vy, vz, w }
  └── SimMass     { mass, inv_mass }
```

### NewtonDynamics

```cpp
void AddTerm(std::unique_ptr<IDynamicsTerm> term);

// Initialize with external buffer handles for bind group caching
void Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                WGPUBuffer mass_buffer, uint32 workgroup_size = 64);

// Dispatch Newton iterations using cached bind groups
void Solve(WGPUCommandEncoder encoder, float32 dt,
           uint32 newton_iterations = 1, uint32 cg_iterations = 30);

void SetGravity(float32 gx, float32 gy, float32 gz);
WGPUBuffer GetDVTotalBuffer() const;
WGPUBuffer GetXOldBuffer() const;
WGPUBuffer GetParamsBuffer() const;
uint64 GetParamsSize() const;
uint64 GetVec4BufferSize() const;
void Shutdown();
```

## Shaders (`assets/shaders/ext_newton/`)

| Shader | Purpose |
|--------|---------|
| `update_velocity.wgsl` | Apply velocity delta: `v = (v + dv_total) * damping` |
| `update_position.wgsl` | Integrate positions: `pos = x_old + v * dt` |
| `newton_init.wgsl` | Initialize Newton iteration (save x_old, zero dv_total) |
| `newton_predict_pos.wgsl` | Predict positions from velocities |
| `newton_accumulate_dv.wgsl` | Accumulate CG solution into dv_total |
| `clear_forces.wgsl` | Zero force buffer |
| `assemble_rhs.wgsl` | Assemble RHS for CG solve |
| `cg_spmv.wgsl` | Sparse matrix-vector product |

All shaders import `core_simulate/header/solver_params.wgsl` (root-relative). CG solver shaders are in `assets/shaders/core_simulate/`. Dynamics term shaders (inertia, gravity, springs, area) are in `assets/shaders/ext_dynamics/`.
