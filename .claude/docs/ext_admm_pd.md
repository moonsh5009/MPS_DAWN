# ext_admm_pd

> ADMM (Alternating Direction Method of Multipliers) PD solver with CG inner solve (Overby 2017). Uses IProjectiveTerm implementations from ext_pd_term.

## Module Structure

```
extensions/ext_admm_pd/
├── CMakeLists.txt                       # STATIC library → mps::ext_admm_pd (depends: mps::core_system, mps::ext_dynamics, mps::ext_newton, mps::ext_pd_term)
├── admm_system_config.h                 # ADMMSystemConfig component (ADMM/CG iterations + constraint entity refs)
├── admm_extension.h / .cpp              # ADMMExtension (IExtension) — registers simulator only
├── admm_dynamics.h / .cpp               # ADMMDynamics solver (ADMM + CG inner solve)
└── admm_system_simulator.h / .cpp       # ADMMSystemSimulator (ISimulator) — discovers terms, runs solver, integrates
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `ADMMSystemConfig` | `admm_system_config.h` | Host-only config: admm_iterations, cg_iterations, constraint entity refs |
| `ADMMDynamics` | `admm_dynamics.h` | ADMM PD solver with CG inner solve and pluggable IProjectiveTerm |
| `ADMMDynamics::SpMVOperator` | `admm_dynamics.h` | Internal ISpMVOperator for the constant PD system matrix (CSR SpMV) |
| `ADMMExtension` | `admm_extension.h` | IExtension: registers ADMMSystemSimulator only (no term providers) |
| `ADMMSystemSimulator` | `admm_system_simulator.h` | ISimulator: discovers PD terms, runs ADMMDynamics, integrates velocity/position |

## API

### ADMMExtension

```cpp
explicit ADMMExtension(mps::system::System& system);
const std::string& GetName() const override;       // "ext_admm_pd"
void Register(mps::system::System& system) override;
```

`Register()` only calls `AddSimulator(ADMMSystemSimulator)`. Term providers are registered by ext_pd_term.

### ADMMSystemSimulator

```cpp
explicit ADMMSystemSimulator(mps::system::System& system);
~ADMMSystemSimulator() override;
const std::string& GetName() const override;       // "ADMMSystemSimulator"
void Initialize() override;
void Update() override;
void Shutdown() override;
void OnDatabaseChanged() override;  // Signature-based topology change detection + reinit
```

`Initialize()` flow:
1. Query Database for entities with `ADMMSystemConfig`
2. Read `constraint_entities[]` array from first config
3. For each constraint entity, call `System::FindAllPDTermProviders()` → `provider->CreateTerm()`
4. Get GPU buffer handles (pos, vel, mass) from DeviceDB; physics buffer from DeviceDB singleton
5. Call `ADMMDynamics::Initialize(node_count, edges, faces, physics_buf, physics_sz, pos, vel, mass)`
6. Create update_velocity and update_position compute pipelines (from `ext_pd_common/`)
7. Cache velocity/position update bind groups
8. Cache `TopologySignature` for change detection

`Update()` flow:
1. Create command encoder
2. Copy-in (scoped mode: global → local buffers)
3. `ADMMDynamics::Solve(encoder)` → ADMM outer loop with CG inner solve
4. Dispatch velocity update: `v = (q_curr - x_old) / dt * damping`
5. Dispatch position update: `pos = x_old + v * dt`
6. Copy-out (scoped mode: local → global buffers)

### ADMMSystemConfig (host-only)

```cpp
struct ADMMSystemConfig {
    static constexpr uint32 MAX_CONSTRAINTS = 8;
    uint32 admm_iterations    = 20;    // Outer ADMM iterations
    uint32 cg_iterations      = 10;    // Inner CG iterations per ADMM step
    float32 penalty_weight    = 0.0f;  // ADMM penalty ρ (0 = auto: M_avg/dt²)
    uint32 constraint_count   = 0;
    uint32 constraint_entities[MAX_CONSTRAINTS] = {};
    uint32 mesh_entity        = database::kInvalidEntity;  // kInvalidEntity = global, valid entity = scoped
    uint32 padding[2]         = {};
};
```

### ADMMDynamics

```cpp
void AddTerm(std::unique_ptr<IProjectiveTerm> term);
void SetADMMIterations(uint32 iterations);
void SetCGIterations(uint32 iterations);
void SetPenaltyWeight(float32 rho);

void Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                WGPUBuffer physics_buffer, uint64 physics_size,
                WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                WGPUBuffer mass_buffer, uint32 workgroup_size = 64);

void Solve(WGPUCommandEncoder encoder);

WGPUBuffer GetQCurrBuffer() const;
WGPUBuffer GetXOldBuffer() const;
WGPUBuffer GetParamsBuffer() const;
uint64 GetParamsSize() const;
uint64 GetVec4BufferSize() const;
void Shutdown();
```

### ADMMDynamics::Solve() — Core Loop

```
1. x_old = positions
2. s = x_old + dt*v + dt²*g
3. q = s (initial guess)
4. for each term: term->ResetDual(encoder)   // z=S*s, u=0

for k = 0..admm_iterations-1:
    // Global step: solve A*q = rhs via CG
    Clear CG RHS
    rhs += (M/dt²) * s                       // mass_rhs shader
    for each term: term->AssembleADMMRHS()    // rhs += w*S^T*(z-u)
    CGSolver::Solve(encoder, cg_iterations)   // CG: A*x = rhs
    Copy cg_x → q_curr
    Fixup pinned nodes: restore q from s       // pd_fixup_pinned shader

    // Local step
    for each term: term->ProjectLocal()       // z = project(S*q + u)

    // Dual step
    for each term: term->UpdateDual()         // u += S*q - z
```

### SpMVOperator (inner class)

Implements `ISpMVOperator` for the constant PD system matrix (M/dt² + Σ w*S^T*S). Dispatches CSR SpMV via `admm_cg_spmv.wgsl` shader.

```cpp
void PrepareSolve(WGPUBuffer p_buffer, uint64 p_size,
                  WGPUBuffer ap_buffer, uint64 ap_size) override;
void Apply(WGPUCommandEncoder encoder, uint32 workgroup_count) override;
```

## Entity Model

```
ADMM System Entity:
  └── ADMMSystemConfig { admm_iterations=20, cg_iterations=10, constraint_refs: [Entity 200] }

Constraint Entity (mesh entity with constraint data):
  └── Entity 200: SpringConstraintData + AreaConstraintData (from ext_dynamics)

Gravity/damping are read from GlobalPhysicsParams DB singleton (ext_dynamics/global_physics_params.h).
```

## Shaders (`assets/shaders/ext_admm_pd/`)

| Shader | Used by | Purpose |
|--------|---------|---------|
| `admm_cg_spmv.wgsl` | SpMVOperator | CSR SpMV: Ap = D*p + offdiag*p for CG inner solve |

Also uses shared PD shaders from `assets/shaders/ext_pd_common/` (pd_init, pd_predict, pd_copy_vec4, pd_mass_rhs, pd_inertial_lhs, pd_fixup_pinned, pd_update_velocity, pd_update_position) and CG solver shaders from `assets/shaders/core_simulate/`.
