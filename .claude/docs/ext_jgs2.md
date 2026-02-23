# ext_jgs2

> JGS² block Jacobi solver (Lan et al., SIGGRAPH 2025) — per-vertex 3×3 block solve with optional frozen Schur complement correction.

## Module Structure

```
extensions/ext_jgs2/
├── CMakeLists.txt                       # STATIC library → mps::ext_jgs2 (depends: mps::core_system, mps::ext_dynamics)
├── jgs2_system_config.h                 # JGS2SystemConfig component (iterations, correction toggle, constraint entity refs)
├── jgs2_extension.h / .cpp              # JGS2Extension (IExtension) — registers simulator only
├── jgs2_dynamics.h / .cpp               # JGS2Dynamics solver + IJGS2Term interface + JGS2AssemblyContext
├── jgs2_spring_term.h / .cpp            # JGS2SpringTerm (IJGS2Term) — spring gradient + diagonal Hessian accumulation
├── jgs2_system_simulator.h / .cpp       # JGS2SystemSimulator (ISimulator) — discovers constraints, runs solver, integrates
└── jgs2_precompute.h / .cpp             # SchurCorrection (Phase 2) — CPU Cholesky factorization + per-vertex correction Δᵢ
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `JGS2SystemConfig` | `jgs2_system_config.h` | Host-only config: iterations, enable_correction, constraint entity refs |
| `JGS2AssemblyContext` | `jgs2_dynamics.h` | GPU buffer handles passed to IJGS2Term::Initialize for bind group caching |
| `IJGS2Term` | `jgs2_dynamics.h` | Extension-local pluggable term interface: `Initialize()`, `Accumulate()`, `Shutdown()` |
| `JGS2Dynamics` | `jgs2_dynamics.h` | Block Jacobi solver with per-vertex 3×3 solve and optional Schur correction |
| `JGS2SpringParams` | `jgs2_spring_term.h` | GPU uniform: `alignas(16) {stiffness}` — 16 bytes |
| `JGS2SpringTerm` | `jgs2_spring_term.h` | IJGS2Term: spring gradient + diagonal Hessian via GPU atomics |
| `JGS2Extension` | `jgs2_extension.h` | IExtension: registers JGS2SystemSimulator only |
| `JGS2SystemSimulator` | `jgs2_system_simulator.h` | ISimulator: discovers constraints directly, runs JGS2Dynamics, integrates velocity/position |
| `SchurCorrection` | `jgs2_precompute.h` | Phase 2: CPU Cholesky factorization + Schur complement correction precomputation |

## API

### JGS2Extension

```cpp
explicit JGS2Extension(mps::system::System& system);
const std::string& GetName() const override;       // "ext_jgs2"
void Register(mps::system::System& system) override;
```

`Register()` only calls `AddSimulator(JGS2SystemSimulator)`. No term providers registered (JGS2 discovers constraints directly).

### JGS2SystemSimulator

```cpp
explicit JGS2SystemSimulator(mps::system::System& system);
~JGS2SystemSimulator() override;
const std::string& GetName() const override;       // "JGS2SystemSimulator"
void Initialize() override;
void Update() override;
void Shutdown() override;
void OnDatabaseChanged() override;  // Signature-based topology change detection + reinit
```

`Initialize()` flow:
1. Query Database for entities with `JGS2SystemConfig`
2. Read `constraint_entities[]` array from first config
3. For each constraint entity, read `SpringConstraintData` + `SpringEdge` array directly (no provider registry)
4. Create `JGS2SpringTerm` per constraint entity with springs
5. If `enable_correction=true`, run `SchurCorrection::Compute()` and upload to GPU
6. Get GPU buffer handles (pos, vel, mass) from DeviceDB; physics buffer from DeviceDB singleton
7. Call `JGS2Dynamics::Initialize(node_count, edges, faces, physics_buf, physics_sz, pos, vel, mass)`
8. Create update_velocity and update_position compute pipelines (from `ext_pd_common/`)
9. Cache velocity/position update bind groups
10. Cache `TopologySignature` for change detection

`Update()` flow:
1. Create command encoder
2. Copy-in (scoped mode: global → local buffers)
3. `JGS2Dynamics::Solve(encoder)` → block Jacobi iteration loop
4. Dispatch velocity update: `v = (q - x_old) / dt * damping`
5. Dispatch position update: `pos = x_old + v * dt`
6. Copy-out (scoped mode: local → global buffers)

### JGS2SystemConfig (host-only)

```cpp
struct JGS2SystemConfig {
    static constexpr uint32 MAX_CONSTRAINTS = 8;
    uint32 iterations         = 10;
    bool enable_correction    = false;  // Phase 2: frozen Schur complement
    uint32 constraint_count   = 0;
    uint32 constraint_entities[MAX_CONSTRAINTS] = {};
    uint32 mesh_entity        = database::kInvalidEntity;
    uint32 padding[3]         = {};
};
```

### JGS2AssemblyContext

```cpp
struct JGS2AssemblyContext {
    WGPUBuffer params_buffer;        // solver params uniform
    WGPUBuffer q_buffer;             // current iterate q (read)
    WGPUBuffer gradient_buffer;      // per-vertex gradient N×4 (atomic u32, read_write)
    WGPUBuffer hessian_diag_buffer;  // per-vertex 3×3 Hessian N×9 (atomic u32, read_write)
    uint32 node_count, edge_count, workgroup_size;
    uint64 params_size;
};
```

### IJGS2Term

```cpp
virtual const std::string& GetName() const = 0;
virtual void Initialize(const JGS2AssemblyContext& ctx) = 0;
virtual void Accumulate(WGPUCommandEncoder encoder) = 0;
virtual void Shutdown() = 0;
```

### JGS2Dynamics

```cpp
void AddTerm(std::unique_ptr<IJGS2Term> term);
void SetIterations(uint32 iterations);
void EnableCorrection(bool enable);

void Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                WGPUBuffer physics_buffer, uint64 physics_size,
                WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                WGPUBuffer mass_buffer, uint32 workgroup_size = 64);

void Solve(WGPUCommandEncoder encoder);
void UploadCorrection(const std::vector<float32>& correction_data);

WGPUBuffer GetQBuffer() const;
WGPUBuffer GetXOldBuffer() const;
WGPUBuffer GetParamsBuffer() const;
uint64 GetParamsSize() const;
uint64 GetVec4BufferSize() const;
void Shutdown();
```

### JGS2Dynamics::Solve() — Core Loop

```
1. x_old = positions          (pd_init)
2. s = x_old + dt*v + dt²*g   (pd_predict)
3. q = s                       (pd_copy_vec4, initial guess)

for k = 0..iterations-1:
    a. Dispatch jgs2_accum_inertia   — atomicStore gradient + Hessian (inertia)
    b. For each term: term->Accumulate()  — atomicAddFloat spring grad + Hess
    c. Dispatch jgs2_local_solve     — δx = -(H+Δ)⁻¹·g, q += δx
```

### SchurCorrection (Phase 2)

```cpp
static std::vector<float32> Compute(
    uint32 node_count,
    const std::vector<float32>& host_positions,  // N×3
    const std::vector<float32>& host_masses,      // N
    const std::vector<std::pair<uint32, uint32>>& edges,
    const std::vector<float32>& rest_lengths,
    float32 stiffness, float32 dt);
```

Algorithm (CPU, O(N³/3) factorization + O(3N³) column solves):
1. Assemble dense rest-pose Hessian H̄ (3N×3N): inertia + spring contributions
2. Cholesky factorization: H̄ = LLᵀ (double precision)
3. For each free vertex i, solve 3 systems H̄·x = eₖ (k=3i,3i+1,3i+2) via forward/back substitution
4. Extract per-vertex (H̄⁻¹)ᵢᵢ from solution columns
5. Compute Sᵢ = ((H̄⁻¹)ᵢᵢ)⁻¹
6. Compute Δᵢ = Sᵢ - H̄ᵢᵢ
7. Return N×9 correction floats

## Entity Model

```
JGS2 System Entity:
  └── JGS2SystemConfig { iterations=10, enable_correction=false, constraint_refs: [Entity 200] }

Constraint Entity (mesh entity with constraint data):
  └── Entity 200: SpringConstraintData (from ext_dynamics)
```

## GPU Buffer Layout (JGS2Dynamics)

| Buffer | Type | Size | Purpose |
|--------|------|------|---------|
| `params_buffer_` | Uniform | 32B | SolverParams (node/edge/face counts) |
| `x_old_buffer_` | Storage | N×16B | Saved positions (vec4f) |
| `s_buffer_` | Storage | N×16B | Predicted positions (vec4f) |
| `q_buffer_` | Storage RW | N×16B | Current iterate (vec4f) |
| `gradient_buffer_` | Storage RW | N×16B | Per-vertex gradient (atomic u32) |
| `hessian_diag_buffer_` | Storage RW | N×36B | Per-vertex 3×3 Hessian (atomic u32) |
| `correction_buffer_` | Storage | N×36B | Precomputed Δᵢ (zeros = Phase 1) |

## Shaders (`assets/shaders/ext_jgs2/`)

| Shader | Used by | Purpose |
|--------|---------|---------|
| `jgs2_accum_inertia.wgsl` | JGS2Dynamics | Per-vertex: gradient = M/dt²·(q-s), Hessian = M/dt²·I₃ (atomicStore) |
| `jgs2_accum_spring.wgsl` | JGS2SpringTerm | Per-edge: spring gradient + PSD diagonal Hessian (atomicAddFloat) |
| `jgs2_local_solve.wgsl` | JGS2Dynamics | Per-vertex: 3×3 cofactor inverse, δx = -(H+Δ)⁻¹·g, q += δx |

Also uses shared PD shaders from `assets/shaders/ext_pd_common/` (pd_init, pd_predict, pd_copy_vec4, pd_update_velocity, pd_update_position).
