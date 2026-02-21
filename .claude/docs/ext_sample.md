# ext_sample

> Minimal reference extension — demonstrates the extension system with simple point particles.

## Module Structure

```
extensions/ext_sample/
├── CMakeLists.txt              # STATIC library → mps::ext_sample (depends: mps::core_system)
├── sample_components.h         # ECS components: SampleTransform, SampleVelocity (16-byte POD)
├── sample_extension.h / .cpp   # SampleExtension (IExtension) — self-contained registration
├── sample_simulator.h / .cpp   # SampleSimulator (ISimulator) — simple update logic
└── sample_renderer.h / .cpp    # SampleRenderer (IObjectRenderer) — point rendering
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `SampleTransform` | `sample_components.h` | `{x, y, z}` — host-only |
| `SampleVelocity` | `sample_components.h` | `{vx, vy, vz}` — host-only |
| `SampleExtension` | `sample_extension.h` | IExtension: registers components, creates entities, adds simulator + renderer |
| `SampleSimulator` | `sample_simulator.h` | ISimulator: simple velocity-based update |
| `SampleRenderer` | `sample_renderer.h` | IObjectRenderer: point rendering |

## API

### SampleExtension

```cpp
explicit SampleExtension(mps::system::System& system);
const std::string& GetName() const override;
void Register(mps::system::System& system) override;
```

### SampleSimulator

```cpp
explicit SampleSimulator(mps::system::System& system);
const std::string& GetName() const override;
void Initialize() override;
void Update(mps::float32 dt) override;
```

### SampleRenderer

```cpp
explicit SampleRenderer(mps::system::System& system);
const std::string& GetName() const override;
void Initialize(mps::render::RenderEngine& engine) override;
void Render(mps::render::RenderEngine& engine, WGPURenderPassEncoder pass) override;
void Shutdown() override;
mps::int32 GetOrder() const override;
```

## Shaders (`assets/shaders/ext_sample/`)

| Shader | Purpose |
|--------|---------|
| `point_vert.wgsl` | Point vertex shader |
| `point_frag.wgsl` | Point fragment shader |
