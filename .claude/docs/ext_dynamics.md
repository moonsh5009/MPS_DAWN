# ext_dynamics

> Shared constraint types, config components, and constraint builder — used by both Newton (ext_newton) and PD (ext_pd) solvers.

## Module Structure

```
extensions/ext_dynamics/
├── CMakeLists.txt                   # STATIC library → mps::ext_dynamics (depends: mps::core_system)
├── dynamics_extension.h / .cpp      # DynamicsExtension (IExtension) — registers GPU arrays
├── constraint_builder.h / .cpp      # BuildSpringConstraints, BuildAreaConstraints — topology from mesh faces
├── spring_constraint.h              # SpringConstraintData component (stiffness config)
├── spring_types.h                   # SpringEdge, EdgeCSRMapping (GPU-compatible topology)
├── area_constraint.h                # AreaConstraintData component (stiffness config)
├── area_types.h                     # AreaTriangle, FaceCSRMapping (GPU-compatible topology)
└── global_physics_params.h          # GlobalPhysicsParams (DB singleton), PhysicsParamsGPU (GPU uniform), ToGPU()
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `DynamicsExtension` | `dynamics_extension.h` | IExtension: registers GlobalPhysicsParams singleton + GPU arrays (SimPosition, SimVelocity, SimMass, SpringEdge, AreaTriangle) |
| `SpringConstraintData` | `spring_constraint.h` | Host-only config: `{stiffness}` — uniform spring stiffness |
| `SpringEdge` | `spring_types.h` | `{n0, n1, rest_length}` — 12 bytes, GPU-compatible |
| `EdgeCSRMapping` | `spring_types.h` | `{block_ab, block_ba, block_aa, block_bb}` — CSR write indices |
| `AreaConstraintData` | `area_constraint.h` | Host-only config: `{stiffness}` |
| `AreaTriangle` | `area_types.h` | `{n0, n1, n2, rest_area, dm_inv_00..11}` — 32 bytes, GPU-compatible |
| `FaceCSRMapping` | `area_types.h` | `{csr_01..21}` — 24 bytes, CSR write indices for triangle edges |
| `GlobalPhysicsParams` | `global_physics_params.h` | Host-side DB singleton: `{dt, gravity, damping}` |
| `PhysicsParamsGPU` | `global_physics_params.h` | GPU uniform (binding 0): `alignas(16) {dt, gravity_xyz, damping, inv_dt, dt_sq, inv_dt_sq}` — 32 bytes |

## API

### BuildSpringConstraints (constraint_builder.h)

```cpp
mps::uint32 BuildSpringConstraints(mps::database::Database& db,
                                   mps::database::Entity mesh_entity,
                                   mps::float32 stiffness);
```

Reads `SimPosition` + `MeshFace` arrays from the mesh entity, extracts unique edges with rest lengths. Writes `SpringEdge` array, adds `SpringConstraintData` component, and updates `MeshComponent::edge_count`. Returns edge count. Must be called inside a `Transact` block.

### BuildAreaConstraints (constraint_builder.h)

```cpp
mps::uint32 BuildAreaConstraints(mps::database::Database& db,
                                 mps::database::Entity mesh_entity,
                                 mps::float32 stiffness);
```

Reads `SimPosition` + `MeshFace` arrays from the mesh entity, computes area triangles with rest area and Dm_inv. Writes `AreaTriangle` array and adds `AreaConstraintData` component. Returns triangle count. Must be called inside a `Transact` block.

### GlobalPhysicsParams (global_physics_params.h)

```cpp
namespace mps::simulate {

// Host-side singleton (stored in Database via SetSingleton)
struct GlobalPhysicsParams {
    float32 dt        = 1.0f / 60.0f;
    util::vec3 gravity = {0.0f, -9.81f, 0.0f};
    float32 damping   = 0.999f;
};

// GPU-side uniform (binding 0, managed by DeviceDB singleton)
struct alignas(16) PhysicsParamsGPU {
    float32 dt, gravity_x, gravity_y, gravity_z;
    float32 damping, inv_dt, dt_sq, inv_dt_sq;
};

inline PhysicsParamsGPU ToGPU(const GlobalPhysicsParams& p);
}
```

Registered in `DynamicsExtension::Register()` via `DeviceDB::RegisterSingleton<GlobalPhysicsParams, PhysicsParamsGPU>(ToGPU)`. All solvers bind the resulting uniform buffer at binding 0.

### DynamicsExtension

```cpp
explicit DynamicsExtension(mps::system::System& system);
const std::string& GetName() const override;       // "ext_dynamics"
void Register(mps::system::System& system) override;
```

`Register()` does:
1. `RegisterSingleton<GlobalPhysicsParams, PhysicsParamsGPU>(ToGPU)` — singleton uniform
2. `RegisterArray<SimPosition>(Vertex)` — GPU-synced
3. `RegisterArray<SimVelocity>()` — GPU-synced
4. `RegisterArray<SimMass>()` — GPU-synced
5. `RegisterIndexedArray<SpringEdge, SimPosition>(offset_fn)` — GPU-synced, auto-offsets node indices
6. `RegisterIndexedArray<AreaTriangle, SimPosition>(offset_fn)` — GPU-synced, auto-offsets node indices

## Entity Model

```
Mesh Entity (also serves as constraint entity):
  ├── SpringConstraintData { stiffness }    (config component, added by BuildSpringConstraints)
  ├── AreaConstraintData { stiffness }      (config component, added by BuildAreaConstraints)
  ├── ArrayStorage<SpringEdge> [...]        (edge topology, written by BuildSpringConstraints)
  └── ArrayStorage<AreaTriangle> [...]      (triangle topology, written by BuildAreaConstraints)
```

## Shaders

None — ext_dynamics contains only host-side data types and constraint builder. Physics shaders live in `assets/shaders/ext_newton/` and `assets/shaders/ext_pd/`.
