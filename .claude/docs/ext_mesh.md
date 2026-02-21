# ext_mesh

> Mesh rendering and normal computation — post-processing and indexed triangle rendering for simulation meshes.

## Module Structure

```
extensions/ext_mesh/
├── CMakeLists.txt                    # STATIC library → mps::ext_mesh (depends: mps::core_system)
├── mesh_types.h                      # MeshFace, FixedVertex (GPU-compatible topology types)
├── mesh_component.h                  # MeshComponent (host-only metadata: vertex/face/edge counts)
├── mesh_generator.h / .cpp           # CreateGrid, ImportOBJ, PinVertices, UnpinVertices (mesh factories)
├── mesh_extension.h / .cpp           # MeshExtension (IExtension) — registers post-processor + renderer
├── mesh_post_processor.h / .cpp      # MeshPostProcessor (ISimulator) — vertex normal computation
├── mesh_renderer.h / .cpp            # MeshRenderer (IObjectRenderer) — indexed triangle rendering
└── normal_computer.h / .cpp          # NormalComputer + NormalParams — GPU normal computation
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `MeshFace` | `mesh_types.h` | `alignas(16) {n0, n1, n2}` — 16 bytes, GPU-compatible triangle face |
| `FixedVertex` | `mesh_types.h` | `{vertex_index, original_mass, original_inv_mass}` — host-only |
| `MeshComponent` | `mesh_component.h` | Host-only metadata: `{vertex_count, face_count, edge_count}` |
| `MeshExtension` | `mesh_extension.h` | IExtension: registers MeshPostProcessor + MeshRenderer |
| `MeshPostProcessor` | `mesh_post_processor.h` | ISimulator: runs NormalComputer, owns face topology GPU buffers |
| `MeshRenderer` | `mesh_renderer.h` | IObjectRenderer: indexed triangle mesh rendering with lit shading |
| `NormalParams` | `normal_computer.h` | `alignas(16) {node_count, face_count}` — 16 bytes, GPU uniform |
| `NormalComputer` | `normal_computer.h` | GPU vertex normal computation (fixed-point i32 atomic scatter + normalize) |
| `MeshResult` | `mesh_generator.h` | Return type for mesh factories: `{mesh_entity, node_count, face_count}` |

## API

### Mesh Generators (mesh_generator.h)

```cpp
struct MeshResult {
    database::Entity mesh_entity = database::kInvalidEntity;
    uint32 node_count = 0;
    uint32 face_count = 0;
};

// Create a grid mesh on XZ plane at Y=height_offset.
// Adds SimPosition, SimVelocity, SimMass (area-weighted), MeshFace, MeshComponent.
MeshResult CreateGrid(database::Database& db,
                      uint32 width, uint32 height, float32 spacing,
                      float32 height_offset = 3.0f);

// Import a triangle mesh from OBJ file (filename relative to assets/objs/).
// Quads are automatically triangulated.
MeshResult ImportOBJ(database::Database& db,
                     const std::string& filename,
                     float32 scale = 1.0f);

// Pin vertices: sets mass=9999999, inv_mass=0, saves original mass in FixedVertex.
void PinVertices(database::Database& db, database::Entity mesh_entity,
                 const std::vector<uint32>& vertex_indices);

// Unpin vertices: restores original mass from FixedVertex.
void UnpinVertices(database::Database& db, database::Entity mesh_entity,
                   const std::vector<uint32>& vertex_indices);
```

All functions must be called inside a `Transact` block.

### MeshExtension

```cpp
explicit MeshExtension(mps::system::System& system);
const std::string& GetName() const override;       // "ext_mesh"
void Register(mps::system::System& system) override;
```

`Register()` does:
1. `AddSimulator(MeshPostProcessor)` — runs after NewtonSystemSimulator
2. `AddRenderer(MeshRenderer)` — receives reference to MeshPostProcessor

### MeshPostProcessor

```cpp
explicit MeshPostProcessor(mps::system::System& system);
~MeshPostProcessor() override;
const std::string& GetName() const override;       // "MeshPostProcessor"
void Initialize() override;
void Update(mps::float32 dt) override;
void Shutdown() override;
void OnDatabaseChanged() override;  // Node/face count change detection + reinit

WGPUBuffer GetNormalBuffer() const;
WGPUBuffer GetIndexBuffer() const;
mps::uint32 GetFaceCount() const;
```

`Initialize()` reads `MeshComponent` metadata and `ArrayStorage<MeshFace>` from Database, creates `NormalComputer`, face buffer, and face index buffer.

`Update(dt)` dispatches `NormalComputer::Compute()` on the current `SimPosition` GPU buffer. Must run after NewtonSystemSimulator updates positions.

`OnDatabaseChanged()` compares current node count and face count against cached values. Calls `Shutdown()` + `Initialize()` on mismatch.

### MeshRenderer

```cpp
MeshRenderer(mps::system::System& system, MeshPostProcessor& post_processor);
const std::string& GetName() const override;       // "MeshRenderer"
void Initialize(mps::render::RenderEngine& engine) override;
void Render(mps::render::RenderEngine& engine, WGPURenderPassEncoder pass) override;
void Shutdown() override;
mps::int32 GetOrder() const override;
```

Gets position buffer from `system.GetDeviceBuffer<SimPosition>()`, normal/index buffers from `MeshPostProcessor`.

### NormalComputer

```cpp
// ext_mesh namespace
struct alignas(16) NormalParams {
    uint32 node_count = 0;
    uint32 face_count = 0;
};

void Initialize(uint32 node_count, uint32 face_count, uint32 workgroup_size = 64);
void Compute(WGPUCommandEncoder encoder,
             WGPUBuffer position_buffer, uint64 position_size,
             WGPUBuffer face_buffer, uint64 face_size);
WGPUBuffer GetNormalBuffer() const;
void Shutdown();
```

Owns its `NormalParams` uniform buffer internally. Caller only provides position and face buffers.

## Entity Model

```
Mesh Entity:
  ├── MeshComponent { vertex_count, face_count, edge_count }  (metadata)
  ├── ArrayStorage<SimPosition> [...]                          (vertex positions)
  ├── ArrayStorage<SimVelocity> [...]                          (vertex velocities)
  ├── ArrayStorage<SimMass> [...]                              (vertex masses)
  └── ArrayStorage<MeshFace> [...]                             (triangle topology)
```

## Shaders (`assets/shaders/ext_mesh/`)

| Shader | Used by | Purpose |
|--------|---------|---------|
| `mesh_vert.wgsl` | MeshRenderer | Vertex shader (position + normal + camera transform) |
| `mesh_frag.wgsl` | MeshRenderer | Fragment shader (lit Blinn-Phong shading) |
| `clear_normals.wgsl` | NormalComputer | Zero normal atomic buffer |
| `normals_scatter.wgsl` | NormalComputer | Scatter face normals to vertices (fixed-point i32 atomics) |
| `normals_normalize.wgsl` | NormalComputer | Normalize accumulated normals to unit vec4f |

Header includes (`assets/shaders/ext_mesh/header/`): `normal_params.wgsl` (NormalParams struct — `node_count`, `face_count`).
