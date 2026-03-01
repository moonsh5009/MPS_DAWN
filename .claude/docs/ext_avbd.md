# ext_avbd

> AVBD (Vertex Block Descent) solver — GPU-based per-vertex block coordinate descent with graph coloring for parallel Gauss-Seidel convergence. Based on He Chen et al., SIGGRAPH 2024.

## Module Structure

```
extensions/ext_avbd/
├── CMakeLists.txt                    # STATIC library → mps::ext_avbd (depends: mps::core_system, mps::ext_dynamics, mps::ext_mesh)
├── avbd_extension.h / .cpp           # AVBDExtension (IExtension) — registers AVBDSystemSimulator
├── avbd_system_config.h              # AVBDSystemConfig (host-only config component)
├── avbd_system_simulator.h / .cpp    # AVBDSystemSimulator (ISimulator) — orchestrates VBD solve with scoped mode
├── avbd_term.h                       # IAVBDTerm interface + AVBDTermContext (extension-local term plugin)
├── avbd_spring_term.h / .cpp         # AVBDSpringTerm (IAVBDTerm) — spring energy gradient + Hessian accumulation
├── vbd_dynamics.h / .cpp             # VBDDynamics — VBD solver core (buffers, pipelines, solve loop)
└── graph_coloring.h / .cpp           # GraphColoring — GPU CSR adjacency build + greedy first-fit coloring
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `AVBDExtension` | `avbd_extension.h` | IExtension: registers AVBDSystemSimulator |
| `AVBDSystemConfig` | `avbd_system_config.h` | Host-only config: `{avbd_iterations, mesh_entity, constraint_count, aa_window, constraint_entities[4]}` — 32 bytes |
| `AVBDSystemSimulator` | `avbd_system_simulator.h` | ISimulator: scoped mode, graph coloring, VBD solve, velocity/position integration |
| `IAVBDTerm` | `avbd_term.h` | Extension-local term interface: `Initialize(ctx)`, `Accumulate(encoder)`, `Shutdown()` |
| `AVBDTermContext` | `avbd_term.h` | Context for term init: q/gradient/hessian buffers + physics/solver uniforms + sizes |
| `AVBDSpringTerm` | `avbd_spring_term.h` | Spring energy term: edge-centric dispatch, atomicAddFloat to gradient (N×4) + hessian_diag (N×9) |
| `AAParams` | `vbd_dynamics.h` | `alignas(16) {m_k, node_count, num_pairs, partial_count, slot_current, ring_size}` — AA uniform (32 bytes) |
| `VBDDynamics` | `vbd_dynamics.h` | VBD solver: working buffers, per-color bind groups, optional Anderson Acceleration (ring buffer + Gram + Cholesky) |
| `GraphColoring` | `graph_coloring.h` | GPU graph coloring: CSR build (prefix sum + fill) + greedy first-fit + color group sort |
| `VBDColorParams` | `vbd_dynamics.h` (private) | `alignas(16) {color_offset, color_vertex_count}` — per-color uniform |
| `ColoringParams` | `graph_coloring.h` (private) | `alignas(16) {node_count, edge_count, scan_count}` — coloring uniform |

## API

### AVBDSystemConfig (avbd_system_config.h)

```cpp
struct AVBDSystemConfig {
    uint32 avbd_iterations    = 10;
    uint32 mesh_entity        = kInvalidEntity;
    uint32 constraint_count   = 0;
    uint32 aa_window          = 5;   // Anderson Acceleration window (0=disabled)
    Entity constraint_entities[4] = {kInvalidEntity, ...};
};
```

### IAVBDTerm (avbd_term.h)

```cpp
struct AVBDTermContext {
    uint32 node_count, edge_count, face_count;
    WGPUBuffer q_buf;              // working positions vec4f
    WGPUBuffer gradient_buf;       // N×4 atomic<u32>
    uint64 gradient_sz;
    WGPUBuffer hessian_buf;        // N×9 atomic<u32>
    uint64 hessian_sz;
    WGPUBuffer physics_buf;        // PhysicsParams uniform
    uint64 physics_sz;
    WGPUBuffer solver_params_buf;  // SolverParams uniform
    uint64 solver_params_sz;
};

class IAVBDTerm {
    virtual const std::string& GetName() const = 0;
    virtual void Initialize(const AVBDTermContext& ctx) = 0;
    virtual void Accumulate(WGPUCommandEncoder encoder) = 0;
    virtual void Shutdown() = 0;
};
```

### AVBDSpringTerm (avbd_spring_term.h)

```cpp
void SetEdgeData(std::span<const ext_dynamics::SpringEdge> edges, float32 stiffness);
void Initialize(const AVBDTermContext& ctx) override;   // creates pipeline + bind group
void Accumulate(WGPUCommandEncoder encoder) override;   // dispatches avbd_accum_spring.wgsl
```

### VBDDynamics (vbd_dynamics.h)

```cpp
void Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                uint32 iterations, uint32 aa_window,
                WGPUBuffer physics_buf, uint64 physics_sz,
                WGPUBuffer pos_buf, WGPUBuffer vel_buf, WGPUBuffer mass_buf,
                uint64 pos_sz, uint64 vel_sz, uint64 mass_sz,
                const std::vector<uint32>& color_offsets,
                WGPUBuffer vertex_order_buf, uint64 vertex_order_sz);
void AddTerm(std::unique_ptr<IAVBDTerm> term);
void Solve(WGPUCommandEncoder encoder);  // full VBD iteration loop
void Shutdown();

WGPUBuffer GetQBuffer() const;
WGPUBuffer GetXOldBuffer() const;
WGPUBuffer GetSolverParamsBuffer() const;
uint64 GetSolverParamsSize() const;
```

### GraphColoring (graph_coloring.h)

```cpp
void Build(uint32 node_count, uint32 edge_count,
           WGPUBuffer edge_buffer, uint64 edge_buffer_size);
void BuildColorGroups();           // CPU readback + counting sort

WGPUBuffer GetColorBuffer() const;       // u32[N]
uint32 GetColorCount() const;            // max_color + 1
WGPUBuffer GetRowPtrBuffer() const;      // u32[N+1]
WGPUBuffer GetColIdxBuffer() const;      // u32[2E]
WGPUBuffer GetVertexOrderBuffer() const; // u32[N] sorted by color
const std::vector<uint32>& GetColorOffsets() const;  // [C+1]
```

### AVBDSystemSimulator (avbd_system_simulator.h)

```cpp
explicit AVBDSystemSimulator(mps::system::System& system);
void Initialize() override;          // scoped mode + coloring + VBD init + term discovery
void Update() override;              // copy-in → solve → velocity → position → copy-out
void Shutdown() override;
void OnDatabaseChanged() override;   // topology signature comparison
```

## Per-Frame Dispatch Flow

```
Command Encoder:
  ┌─ Copy-in (scoped): global → local
  ├─ avbd_init: x_old = pos                              [ceil(N/64)]
  ├─ avbd_predict: s = x_old + dt*v + dt²*g              [ceil(N/64)]
  ├─ avbd_copy_q: q = s                                  [ceil(N/64)]
  ├─ VBD Iteration Loop (I iterations × C colors):
  │   ├─ (AA) avbd_aa_save_x: x_save = q                  [ceil(N/64)]
  │   ├─ avbd_accum_inertia (ALL vertices, atomicStore)    [ceil(N/64)]
  │   ├─ avbd_accum_spring (ALL edges, atomicAddFloat)     [ceil(E/64)]
  │   ├─ avbd_local_solve (color c vertices only)          [ceil(|c|/64)]
  │   ├─ (AA) avbd_aa_store_gf: ring buffer store          [ceil(N/64)]
  │   ├─ (AA, iter≥1) avbd_aa_gram: Gram partial sums     [ceil(N/64)]
  │   ├─ (AA, iter≥1) avbd_aa_solve: Cholesky + coeffs    [1 workgroup]
  │   └─ (AA, iter≥1) avbd_aa_blend: q = Σ c_j·g^{k-j}   [ceil(N/64)]
  ├─ avbd_update_velocity                                 [ceil(N/64)]
  ├─ avbd_update_position                                 [ceil(N/64)]
  └─ Copy-out (scoped): local → global
```

Total dispatches: 5 + 3·I·C (e.g., 10 iter × 4 colors = 125 dispatches/frame).

## Buffer Layout

| Buffer | Size | Type | Purpose |
|--------|------|------|---------|
| `x_old` | N×16B | vec4f | Initial positions |
| `s` | N×16B | vec4f | Predicted positions |
| `q` | N×16B | vec4f | Working iterate (in-place updates) |
| `gradient` | N×16B | 4×atomic<u32> | Energy gradient (float→u32 bitcast) |
| `hessian_diag` | N×36B | 9×atomic<u32> | 3×3 block diagonal Hessian |
| `vertex_order` | N×4B | u32 | Vertices sorted by color |
| `color_params[c]` | 16B | Uniform | Per-color offset/count |
| `solver_params` | 32B | Uniform | SolverParams (node/edge/face counts) |
| `x_save` (AA) | N×16B | vec4f | Snapshot before GS sweep |
| `g_history` (AA) | (m+1)×N×16B | vec4f | Ring buffer of GS outputs |
| `f_history` (AA) | (m+1)×N×16B | vec4f | Ring buffer of residuals |
| `gram_partials` (AA) | P×W×4B | f32 | Partial sums for Gram matrix (P=pairs, W=workgroups) |
| `blend_coeffs` (AA) | (m+1)×4B | f32 | Blending coefficients from Cholesky solve |
| `aa_params[ci]` (AA) | 32B | Uniform | Per-config AAParams (2m+1 configs pre-cached) |

## Shaders (`assets/shaders/ext_avbd/`)

### VBD Solver Shaders (workgroup size 64)

| Shader | Bindings | Purpose |
|--------|----------|---------|
| `avbd_init.wgsl` | 0=SolverParams, 1=pos, 2=x_old | x_old = pos |
| `avbd_predict.wgsl` | 0=Physics, 1=Solver, 2=x_old, 3=vel, 4=mass, 5=s | s = x_old + dt·v + dt²·g (pinned: s = x_old) |
| `avbd_copy_q.wgsl` | 0=SolverParams, 1=s, 2=q | q = s |
| `avbd_accum_inertia.wgsl` | 0=Physics, 1=Solver, 2=q, 3=s, 4=mass, 5=grad, 6=hess | Inertia init via atomicStore |
| `avbd_accum_spring.wgsl` | 0=Physics, 1=Solver, 2=q, 3=edges, 4=spring_params, 5=grad, 6=hess | Spring gradient + PSD Hessian via atomicAddFloat |
| `avbd_local_solve.wgsl` | 0=ColorParams, 1=q, 2=mass, 3=grad, 4=hess, 5=vertex_order | 3×3 cofactor inverse, q += -H⁻¹·g |
| `avbd_update_velocity.wgsl` | 0=Physics, 1=Solver, 2=vel, 3=q, 4=x_old, 5=mass | v = (q-x_old)·inv_dt·damping |
| `avbd_update_position.wgsl` | 0=Solver, 1=pos, 2=q, 3=x_old, 4=mass | pos = q (free), pos = x_old (pinned) |

### Anderson Acceleration Shaders (workgroup size 64)

| Shader | Bindings | Purpose |
|--------|----------|---------|
| `avbd_aa_save_x.wgsl` | 0=SolverParams, 1=q, 2=x_save | x_save = q (snapshot before GS) |
| `avbd_aa_store_gf.wgsl` | 0=SolverParams, 1=AAParams, 2=q, 3=x_save, 4=g_history, 5=f_history | Store g=q, f=q-x_save in ring buffer |
| `avbd_aa_gram.wgsl` | 0=AAParams, 1=f_history, 2=gram_partials | Multi-dot-product reduction (Gram matrix + RHS) |
| `avbd_aa_solve.wgsl` | 0=AAParams, 1=gram_partials, 2=blend_coeffs | Final reduce + Cholesky solve + blend coefficients |
| `avbd_aa_blend.wgsl` | 0=SolverParams, 1=AAParams, 2=g_history, 3=blend_coeffs, 4=q | q = Σ c_j · g^{k-j} weighted blend |

### Graph Coloring Shaders (workgroup size 256)

| Shader | Purpose |
|--------|---------|
| `gc_count_degrees.wgsl` | Count vertex degrees from edge list (atomicAdd) |
| `gc_prefix_sum_local.wgsl` | Blelloch exclusive prefix sum (local workgroup) |
| `gc_prefix_sum_propagate.wgsl` | Propagate block sums back to elements |
| `gc_fill_adjacency.wgsl` | Fill CSR col_idx from edge list (atomic write counters) |
| `gc_color_vertices.wgsl` | Greedy first-fit coloring (64-bit bitmask, iterative) |
| `gc_find_max_color.wgsl` | Reduction to find maximum color value |

Header includes (`assets/shaders/ext_avbd/header/`): `coloring_params.wgsl`, `vbd_params.wgsl`, `aa_params.wgsl`.

## Design Patterns

- **Scoped mode**: Local GPU buffers (pos, vel, mass) with offset-based copy-in/out from global DeviceDB buffers. Same pattern as ext_newton.
- **Direct constraint discovery**: No provider registry. Config's `constraint_entities[]` checked for `SpringConstraintData` component. Term created directly from host DB array data.
- **Atomic accumulation**: Inertia via `atomicStore` (initializes), terms via `atomicAddFloat` (accumulates). Local solve reads as plain `u32` with `bitcast<f32>`.
- **Per-color bind groups**: Each color group has its own uniform buffer (`VBDColorParams`) and bind group for the local solve pipeline. Created once at init time.
- **True Gauss-Seidel**: Re-accumulation per color group ensures position updates from previous colors are reflected in the next color's gradient/Hessian.
- **Anderson Acceleration (AA)**: Type-I AA with window size m (default 5). Ring buffer stores m+1 GS outputs (g) and residuals (f). Per-iteration: Gram matrix ΔF^TΔF via multi-dot workgroup reduction → Cholesky solve → weighted blend of g-history. Pre-cached 2m+1 AAConfig bind groups (warmup + steady-state slot rotations). Disabled when `aa_window=0`.
