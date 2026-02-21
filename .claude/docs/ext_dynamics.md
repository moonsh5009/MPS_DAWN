# ext_dynamics

> All IDynamicsTerm implementations — inertial, gravity, spring (edge-based), and area (triangle-based) forces for Newton-Raphson dynamics.

## Module Structure

```
extensions/ext_dynamics/
├── CMakeLists.txt                   # STATIC library → mps::ext_dynamics (depends: mps::core_system)
├── dynamics_extension.h / .cpp      # DynamicsExtension (IExtension) — registers all term providers
├── constraint_builder.h / .cpp      # BuildConstraintsFromFaces — edge/area topology from mesh faces
├── inertial_term.h / .cpp           # InertialTerm (IDynamicsTerm) — mass diagonal: A_ii += M_i * I3x3
├── gravity_term.h / .cpp            # GravityTerm (IDynamicsTerm) — force: f[i] += mass_i * gravity
├── spring_constraint.h              # SpringConstraintData component (marker for constraint entity)
├── spring_types.h                   # SpringEdge, EdgeCSRMapping (GPU-compatible topology)
├── spring_term_provider.h / .cpp    # SpringTermProvider (IDynamicsTermProvider → SpringTerm)
├── spring_term.h / .cpp             # SpringTerm (IDynamicsTerm) — spring Hessian + force assembly
├── area_constraint.h                # AreaConstraintData component (stiffness config)
├── area_types.h                     # AreaTriangle, FaceCSRMapping (GPU-compatible topology)
├── area_term_provider.h / .cpp      # AreaTermProvider (IDynamicsTermProvider → AreaTerm)
└── area_term.h / .cpp               # AreaTerm (IDynamicsTerm) — area preservation force + full Hessian
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `ConstraintResult` | `constraint_builder.h` | Return type: `{edge_count, area_count}` |
| `DynamicsExtension` | `dynamics_extension.h` | IExtension: registers SpringTermProvider + AreaTermProvider |
| `InertialTerm` | `inertial_term.h` | IDynamicsTerm: writes mass to A diagonal (`A_ii += M_i * I3x3`) |
| `GravityTerm` | `gravity_term.h` | IDynamicsTerm: gravitational force (`force[i] += mass_i * gravity`) |
| `SpringConstraintData` | `spring_constraint.h` | Host-only config: `{stiffness}` — uniform spring stiffness |
| `SpringEdge` | `spring_types.h` | `{n0, n1, rest_length}` — 12 bytes, GPU-compatible |
| `EdgeCSRMapping` | `spring_types.h` | `{block_ab, block_ba, block_aa, block_bb}` — CSR write indices |
| `SpringTermProvider` | `spring_term_provider.h` | IDynamicsTermProvider: creates SpringTerm from ArrayStorage<SpringEdge> |
| `SpringParams` | `spring_term.h` | GPU uniform: `alignas(16) {stiffness}` — 16 bytes |
| `SpringTerm` | `spring_term.h` | IDynamicsTerm: spring Hessian (`-dt²*H`) + force assembly |
| `AreaConstraintData` | `area_constraint.h` | Host-only config: `{stiffness}` |
| `AreaTriangle` | `area_types.h` | `{n0, n1, n2, rest_area, dm_inv_00..11}` — 32 bytes, GPU-compatible |
| `FaceCSRMapping` | `area_types.h` | `{csr_01..21}` — 24 bytes, CSR write indices for triangle edges |
| `AreaTermProvider` | `area_term_provider.h` | IDynamicsTermProvider: creates AreaTerm from ArrayStorage<AreaTriangle> |
| `AreaTerm` | `area_term.h` | IDynamicsTerm: area preservation force + full Hessian (diagonal + off-diagonal CSR) |
| `AreaParams` | `area_term.h` | GPU uniform: `alignas(16) {stiffness, shear_stiffness}` — 16 bytes |

## API

### BuildConstraintsFromFaces (constraint_builder.h)

```cpp
struct ConstraintResult {
    mps::uint32 edge_count = 0;
    mps::uint32 area_count = 0;
};

ConstraintResult BuildConstraintsFromFaces(mps::database::Database& db,
                                           mps::database::Entity mesh_entity);
```

Reads `SimPosition` + `MeshFace` arrays from the mesh entity, extracts unique edges (with rest lengths) and computes area triangles (with rest area + Dm_inv). Writes `SpringEdge` and `AreaTriangle` arrays, updates `MeshComponent::edge_count`. Spring stiffness is configured separately via `SpringConstraintData` on the constraint entity. Must be called inside a `Transact` block.

### DynamicsExtension

```cpp
explicit DynamicsExtension(mps::system::System& system);
const std::string& GetName() const override;       // "ext_dynamics"
void Register(mps::system::System& system) override;
```

`Register()` does:
1. `RegisterTermProvider(SpringConstraintData → SpringTermProvider)`
2. `RegisterTermProvider(AreaConstraintData → AreaTermProvider)`

### InertialTerm

```cpp
InertialTerm() = default;
const std::string& GetName() const override;
void Initialize(const SparsityBuilder& sparsity, const AssemblyContext& ctx) override;
void Assemble(WGPUCommandEncoder encoder) override;
void Shutdown() override;
```

### GravityTerm

```cpp
GravityTerm() = default;
const std::string& GetName() const override;
void Initialize(const SparsityBuilder& sparsity, const AssemblyContext& ctx) override;
void Assemble(WGPUCommandEncoder encoder) override;
void Shutdown() override;
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
SpringTerm(const std::vector<SpringEdge>& edges, mps::float32 stiffness);
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
AreaTerm(const std::vector<AreaTriangle>& triangles, float32 stiffness);
const std::string& GetName() const override;
void DeclareSparsity(SparsityBuilder& builder) override;
void Initialize(const SparsityBuilder& sparsity, const AssemblyContext& ctx) override;
void Assemble(WGPUCommandEncoder encoder) override;
void Shutdown() override;
```

## Entity Model

```
Constraint Entity (spring):
  ├── SpringConstraintData { }        (marker component)
  └── ArrayStorage<SpringEdge> [...]  (edge topology)

Constraint Entity (area):
  ├── AreaConstraintData { stiffness } (config component)
  └── ArrayStorage<AreaTriangle> [...] (triangle topology)
```

## Shaders (`assets/shaders/ext_dynamics/`)

| Shader | Used by | Purpose |
|--------|---------|---------|
| `inertia_assemble.wgsl` | InertialTerm | Write mass to A diagonal: `A_ii += M_i * I3x3` |
| `accumulate_gravity.wgsl` | GravityTerm | Add gravity forces: `force[i] += mass_i * g` |
| `accumulate_springs.wgsl` | SpringTerm | Spring force + Hessian assembly (`-dt²*H` off-diagonal, `+dt²*H` diagonal) |
| `accumulate_area.wgsl` | AreaTerm | Area preservation force + full Hessian (FEM/SVD + ARAP shear) |
