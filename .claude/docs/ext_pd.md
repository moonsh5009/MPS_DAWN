# ext_pd

> Projective Dynamics solver — Chebyshev-accelerated Jacobi iteration with pluggable IProjectiveTerm constraints (Wang 2015).

## Module Structure

```
extensions/ext_pd/
├── CMakeLists.txt                       # STATIC library → mps::ext_pd (depends: mps::core_system, mps::ext_dynamics, mps::ext_newton)
├── pd_dynamics.h / .cpp                 # PDDynamics solver (PD + CSR + Chebyshev Jacobi)
├── pd_system_config.h                   # PDSystemConfig component (solver params + constraint entity refs)
├── pd_extension.h / .cpp               # PDExtension (IExtension) — registers config, providers, simulator
├── pd_system_simulator.h / .cpp         # PDSystemSimulator (ISimulator) — discovers terms, runs solver, integrates
├── pd_spring_term.h / .cpp              # PDSpringTerm (IProjectiveTerm) — spring distance constraint
├── pd_spring_term_provider.h / .cpp     # PDSpringTermProvider (IProjectiveTermProvider → PDSpringTerm)
├── pd_area_term.h / .cpp                # PDAreaTerm (IProjectiveTerm) — ARAP area constraint
└── pd_area_term_provider.h / .cpp       # PDAreaTermProvider (IProjectiveTermProvider → PDAreaTerm)
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `PDSystemConfig` | `pd_system_config.h` | Host-only config: iterations, chebyshev_rho (0=auto), constraint entity refs (64 bytes) |
| `JacobiParams` | `pd_dynamics.h` | GPU uniform: `alignas(16) {omega, is_first_step}` — 16 bytes |
| `PDDynamics` | `pd_dynamics.h` | PD solver with Chebyshev-accelerated Jacobi and pluggable IProjectiveTerm |
| `PDExtension` | `pd_extension.h` | IExtension: registers PDSystemConfig, PD term providers, PDSystemSimulator |
| `PDSystemSimulator` | `pd_system_simulator.h` | ISimulator: discovers PD terms, runs PDDynamics, integrates velocity/position |
| `PDSpringTerm` | `pd_spring_term.h` | IProjectiveTerm: spring distance constraint (LHS + local projection + RHS) |
| `PDSpringTermProvider` | `pd_spring_term_provider.h` | IProjectiveTermProvider: creates PDSpringTerm from SpringConstraintData |
| `PDAreaTerm` | `pd_area_term.h` | IProjectiveTerm: ARAP area constraint (LHS + local projection + RHS) |
| `PDAreaTermProvider` | `pd_area_term_provider.h` | IProjectiveTermProvider: creates PDAreaTerm from AreaConstraintData |

## API

### PDExtension

```cpp
explicit PDExtension(mps::system::System& system);
const std::string& GetName() const override;       // "ext_pd"
void Register(mps::system::System& system) override;
```

`Register()` does:
1. `RegisterPDTermProvider(SpringConstraintData → PDSpringTermProvider)`
2. `RegisterPDTermProvider(AreaConstraintData → PDAreaTermProvider)`
3. `AddSimulator(PDSystemSimulator)`

### PDSystemSimulator

```cpp
explicit PDSystemSimulator(mps::system::System& system);
~PDSystemSimulator() override;
const std::string& GetName() const override;       // "PDSystemSimulator"
void Initialize() override;
void Update() override;
void Shutdown() override;
void OnDatabaseChanged() override;  // Signature-based topology change detection + reinit
```

`Initialize()` flow:
1. Query Database for entities with `PDSystemConfig`
2. Read `constraint_entities[]` array from first config
3. For each constraint entity, call `System::FindAllPDTermProviders()` → `provider->CreateTerm()`
4. Get GPU buffer handles (pos, vel, mass) from DeviceDB; physics buffer from DeviceDB singleton
5. Call `PDDynamics::Initialize(node_count, edges, faces, physics_buf, physics_sz, pos, vel, mass)`
6. Create update_velocity and update_position compute pipelines
7. Cache velocity/position update bind groups
8. Cache `TopologySignature` for change detection

`Update()` flow:
1. Create command encoder
2. Copy-in (scoped mode: global → local buffers)
3. `PDDynamics::Solve(encoder)` → single fused loop
4. Dispatch velocity update: `v = (q_curr - x_old) / dt * damping`
5. Dispatch position copy: `pos = q_curr`
6. Copy-out (scoped mode: local → global buffers)

### PDSystemConfig (64 bytes, host-only)

```cpp
struct PDSystemConfig {
    static constexpr uint32 MAX_CONSTRAINTS = 8;
    uint32 iterations         = 20;     // Wang 2015 single fused loop iterations
    float32 chebyshev_rho     = 0.0f;   // 0 = auto-compute from LHS; >0 = manual override
    uint32 constraint_count   = 0;
    uint32 constraint_entities[MAX_CONSTRAINTS] = {};
    uint32 mesh_entity        = database::kInvalidEntity;  // kInvalidEntity = global, valid entity = scoped
    uint32 padding[4]         = {};
};
```

### PDDynamics

```cpp
void AddTerm(std::unique_ptr<IProjectiveTerm> term);

// Configure solver iterations (call before Initialize or anytime)
void SetIterations(uint32 iterations);
void SetChebyshevRho(float32 rho);

// Initialize with external buffer handles for bind group caching.
// physics_buffer: DeviceDB singleton uniform (binding 0).
void Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                WGPUBuffer physics_buffer, uint64 physics_size,
                WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                WGPUBuffer mass_buffer, uint32 workgroup_size = 64);

// Run PD solver (uses cached bind groups, no per-frame params)
void Solve(WGPUCommandEncoder encoder);

// Adaptive ρ calibration (Wang 2015 gradient decrease rate method).
// Runs pure Jacobi iterations, measures convergence rate, builds Chebyshev params.
// Call before first Solve(). Returns true if calibration was performed.
bool CalibrateRho();
bool IsRhoCalibrated() const;

WGPUBuffer GetQCurrBuffer() const;
WGPUBuffer GetXOldBuffer() const;
WGPUBuffer GetParamsBuffer() const;
uint64 GetParamsSize() const;
uint64 GetVec4BufferSize() const;
void Shutdown();
```

### PDSpringTermProvider

```cpp
std::string_view GetTermName() const override;                       // "PDSpringTermProvider"
bool HasConfig(const Database& db, Entity entity) const override;    // checks SpringConstraintData
std::unique_ptr<IProjectiveTerm> CreateTerm(
    const Database& db, Entity entity, uint32 node_count) override;
void DeclareTopology(uint32& out_edge_count, uint32& out_face_count) override;
void QueryTopology(const Database& db, Entity entity,
                   uint32& out_edge_count, uint32& out_face_count) const override;
```

### PDAreaTermProvider

```cpp
std::string_view GetTermName() const override;                       // "PDAreaTermProvider"
bool HasConfig(const Database& db, Entity entity) const override;    // checks AreaConstraintData
std::unique_ptr<IProjectiveTerm> CreateTerm(
    const Database& db, Entity entity, uint32 node_count) override;
void DeclareTopology(uint32& out_edge_count, uint32& out_face_count) override;
void QueryTopology(const Database& db, Entity entity,
                   uint32& out_edge_count, uint32& out_face_count) const override;
```

### PDSpringTerm

```cpp
PDSpringTerm(const std::vector<ext_dynamics::SpringEdge>& edges, float32 stiffness);
const std::string& GetName() const override;
void DeclareSparsity(SparsityBuilder& builder) override;
void Initialize(const SparsityBuilder& sparsity, const PDAssemblyContext& ctx) override;
void AssembleLHS(WGPUCommandEncoder encoder) override;
void ProjectRHS(WGPUCommandEncoder encoder) override;   // fused local projection + RHS
void Shutdown() override;
```

### PDAreaTerm

```cpp
PDAreaTerm(const std::vector<ext_dynamics::AreaTriangle>& triangles, float32 stiffness);
const std::string& GetName() const override;
void DeclareSparsity(SparsityBuilder& builder) override;
void Initialize(const SparsityBuilder& sparsity, const PDAssemblyContext& ctx) override;
void AssembleLHS(WGPUCommandEncoder encoder) override;
void ProjectRHS(WGPUCommandEncoder encoder) override;   // fused local projection + RHS
void Shutdown() override;
```

## Entity Model

```
PD System Entity:
  └── PDSystemConfig { iterations, chebyshev_rho, constraint_refs: [Entity 200] }

Constraint Entity (mesh entity with constraint data):
  └── Entity 200: SpringConstraintData + AreaConstraintData (from ext_dynamics)

Gravity/damping are read from GlobalPhysicsParams DB singleton (ext_dynamics/global_physics_params.h).
```

## Shaders (`assets/shaders/ext_pd/`)

| Shader | Used by | Purpose |
|--------|---------|---------|
| `pd_init.wgsl` | PDDynamics | Save x_old = positions |
| `pd_predict.wgsl` | PDDynamics | Predict: s = x_old + dt*v + dt²*g |
| `pd_copy_vec4.wgsl` | PDDynamics / PDSystemSimulator | Generic vec4 buffer copy (q←s, prev←curr, rhs clear, pos←q) |
| `pd_mass_rhs.wgsl` | PDDynamics | Mass RHS: rhs += M/dt² * s |
| `pd_inertial_lhs.wgsl` | PDDynamics | Inertial LHS: diag += M/dt² * I |
| `pd_compute_d_inv.wgsl` | PDDynamics | Compute D⁻¹ (inverse of diagonal blocks) |
| `pd_jacobi_step.wgsl` | PDDynamics | Fused off-diagonal SpMV + Jacobi + Chebyshev step |
| `pd_update_velocity.wgsl` | PDSystemSimulator | Velocity from displacement: v = (q - x_old) / dt * damping |
| `pd_update_position.wgsl` | PDSystemSimulator | Copy positions: pos = q_curr |
| `pd_spring_lhs.wgsl` | PDSpringTerm | Spring LHS assembly (w * S^T * S) |
| `pd_spring_project_rhs.wgsl` | PDSpringTerm | Fused spring local projection + RHS assembly |
| `pd_area_lhs.wgsl` | PDAreaTerm | Area LHS assembly |
| `pd_area_project_rhs.wgsl` | PDAreaTerm | Fused area ARAP local projection + RHS assembly |

All shaders import `core_simulate/header/solver_params.wgsl` (root-relative). Shaders that use physics params also import `core_simulate/header/physics_params.wgsl`.
