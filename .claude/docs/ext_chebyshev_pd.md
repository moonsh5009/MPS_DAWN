# ext_chebyshev_pd

> Chebyshev-accelerated Jacobi PD solver (Wang 2015). Uses IProjectiveTerm implementations from ext_pd_term.

## Module Structure

```
extensions/ext_chebyshev_pd/
├── CMakeLists.txt                       # STATIC library → mps::ext_chebyshev_pd (depends: mps::core_system, mps::ext_dynamics, mps::ext_newton, mps::ext_pd_term)
├── pd_system_config.h                   # ChebyshevPDSystemConfig component (solver params + constraint entity refs)
├── pd_extension.h / .cpp               # ChebyshevPDExtension (IExtension) — registers simulator only
├── pd_dynamics.h / .cpp                 # PDDynamics solver (Chebyshev-accelerated Jacobi + CSR)
└── pd_system_simulator.h / .cpp         # ChebyshevPDSystemSimulator (ISimulator) — discovers terms, runs solver, integrates
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `ChebyshevPDSystemConfig` | `pd_system_config.h` | Host-only config: iterations, chebyshev_rho (0=auto), constraint entity refs (64 bytes) |
| `JacobiParams` | `pd_dynamics.h` | GPU uniform: `alignas(16) {omega, is_first_step}` — 16 bytes |
| `PDDynamics` | `pd_dynamics.h` | PD solver with Chebyshev-accelerated Jacobi and pluggable IProjectiveTerm |
| `ChebyshevPDExtension` | `pd_extension.h` | IExtension: registers ChebyshevPDSystemSimulator only (no term providers) |
| `ChebyshevPDSystemSimulator` | `pd_system_simulator.h` | ISimulator: discovers PD terms, runs PDDynamics, integrates velocity/position |

## API

### ChebyshevPDExtension

```cpp
explicit ChebyshevPDExtension(mps::system::System& system);
const std::string& GetName() const override;       // "ext_chebyshev_pd"
void Register(mps::system::System& system) override;
```

`Register()` only calls `AddSimulator(ChebyshevPDSystemSimulator)`. Term providers are registered by ext_pd_term.

### ChebyshevPDSystemSimulator

```cpp
explicit ChebyshevPDSystemSimulator(mps::system::System& system);
~ChebyshevPDSystemSimulator() override;
const std::string& GetName() const override;       // "ChebyshevPDSystemSimulator"
void Initialize() override;
void Update() override;
void Shutdown() override;
void OnDatabaseChanged() override;  // Signature-based topology change detection + reinit
```

`Initialize()` flow:
1. Query Database for entities with `ChebyshevPDSystemConfig`
2. Read `constraint_entities[]` array from first config
3. For each constraint entity, call `System::FindAllPDTermProviders()` → `provider->CreateTerm()`
4. Get GPU buffer handles (pos, vel, mass) from DeviceDB; physics buffer from DeviceDB singleton
5. Call `PDDynamics::Initialize(node_count, edges, faces, physics_buf, physics_sz, pos, vel, mass)`
6. Create update_velocity and update_position compute pipelines (from `ext_pd_common/`)
7. Cache velocity/position update bind groups
8. Cache `TopologySignature` for change detection

`Update()` flow:
1. Create command encoder
2. Copy-in (scoped mode: global → local buffers)
3. CalibrateRho (first frame only)
4. `PDDynamics::Solve(encoder)` → single fused loop with Chebyshev acceleration
5. Dispatch velocity update: `v = (q_curr - x_old) / dt * damping`
6. Dispatch position update: `pos = x_old + v * dt`
7. Copy-out (scoped mode: local → global buffers)

### ChebyshevPDSystemConfig (64 bytes, host-only)

```cpp
struct ChebyshevPDSystemConfig {
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
void SetIterations(uint32 iterations);
void SetChebyshevRho(float32 rho);

void Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                WGPUBuffer physics_buffer, uint64 physics_size,
                WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                WGPUBuffer mass_buffer, uint32 workgroup_size = 64);

void Solve(WGPUCommandEncoder encoder);
bool CalibrateRho();
bool IsRhoCalibrated() const;

WGPUBuffer GetQCurrBuffer() const;
WGPUBuffer GetXOldBuffer() const;
WGPUBuffer GetParamsBuffer() const;
uint64 GetParamsSize() const;
uint64 GetVec4BufferSize() const;
void DebugDump();
void Shutdown();
```

## Shaders

### Shared PD infrastructure (`assets/shaders/ext_pd_common/`)

| Shader | Used by | Purpose |
|--------|---------|---------|
| `pd_init.wgsl` | PDDynamics / ADMMDynamics | Save x_old = positions |
| `pd_predict.wgsl` | PDDynamics / ADMMDynamics | Predict: s = x_old + dt*v + dt²*g |
| `pd_copy_vec4.wgsl` | PDDynamics / ADMMDynamics | Generic vec4 buffer copy |
| `pd_mass_rhs.wgsl` | PDDynamics / ADMMDynamics | Mass RHS: rhs += M/dt² * s |
| `pd_inertial_lhs.wgsl` | PDDynamics / ADMMDynamics | Inertial LHS: diag += M/dt² * I |
| `pd_compute_d_inv.wgsl` | PDDynamics | Compute D⁻¹ (inverse of diagonal blocks) |
| `pd_update_velocity.wgsl` | ChebyshevPDSystemSimulator / ADMMSystemSimulator | Velocity: v = (q - x_old) / dt * damping |
| `pd_update_position.wgsl` | ChebyshevPDSystemSimulator / ADMMSystemSimulator | Position: pos = x_old + v * dt |

### Chebyshev-specific (`assets/shaders/ext_chebyshev_pd/`)

| Shader | Used by | Purpose |
|--------|---------|---------|
| `pd_jacobi_step.wgsl` | PDDynamics | Fused off-diagonal SpMV + Jacobi + Chebyshev step |
