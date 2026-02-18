---
name: system
description: System controller that orchestrates database, simulate, and render. Owns core_system module. Use when implementing or modifying the system coordination layer.
model: opus
memory: project
---

# System Agent

Owns the `core_system` module. Manages the System controller that orchestrates database, simulate, and render modules.

## Module Structure

```
src/core_system/
├── CMakeLists.txt         # STATIC library → mps::core_system (depends: core_util, core_gpu, core_database, core_simulate, core_render)
├── system.h / system.cpp  # System facade (Database + DeviceDB + extension orchestration)
└── extension.h            # IExtension interface
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `System` | `system.h` | Facade: component registration, transactional mutations with auto GPU sync, undo/redo, extension orchestration |
| `IExtension` | `extension.h` | Extension entry point: `Register(System&)` — registers components, simulators, renderers |

## System API

```cpp
class System {
    // Component registration (host + device)
    template<database::Component T>
    void RegisterComponent(gpu::BufferUsage usage, const std::string& label = "");

    // Transactional mutations with auto GPU sync
    void Transact(std::function<void(database::Database&)> fn);

    void Undo();
    void Redo();
    bool CanUndo() const;
    bool CanRedo() const;

    // ECS access
    const database::Database& GetDatabase() const;
    simulate::DeviceDB& GetDeviceDB();

    // GPU buffer access (for render)
    template<database::Component T>
    WGPUBuffer GetDeviceBuffer() const;

    // --- Extension system ---
    void AddExtension(std::unique_ptr<IExtension> extension);
    void AddSimulator(std::unique_ptr<simulate::ISimulator> simulator);
    void AddRenderer(std::unique_ptr<render::IObjectRenderer> renderer);
    void InitializeExtensions(render::RenderEngine& engine);
    void ShutdownExtensions();
    void UpdateSimulators(float32 dt);
    void RenderAll(render::RenderEngine& engine, WGPURenderPassEncoder pass);
};
```

## Extension Flow

```
main.cpp
  ├── system.AddExtension(ext)
  ├── system.InitializeExtensions(engine)
  │     ├── ext->Register(system)         ← components, simulators, renderers
  │     ├── sim->Initialize(db)           ← wrapped in transaction
  │     └── renderer->Initialize(engine)
  └── main loop:
        ├── system.UpdateSimulators(dt)   ← each wrapped in transaction, auto GPU sync
        └── system.RenderAll(engine, pass)← sorted by GetOrder()
```

## Data Flow

```
User code ──Transact()──► Database (host)
                              │
                         Sync() (auto)
                              │
                              ▼
                         DeviceDB (GPU)
                              │
                    GetDeviceBuffer<T>()
                              │
                              ▼
                      core_render (reads handles)
```

## Rules

- Namespace: `mps::system`
- System does NOT own RenderEngine — it passes GPU buffer handles to render
- Every Transact/Undo/Redo auto-syncs dirty data to GPU via DeviceDB::Sync()
- core_render has NO dependency on core_database or core_simulate — only reads WGPUBuffer handles
- Extensions link `mps::core_system` which transitively provides all core modules
