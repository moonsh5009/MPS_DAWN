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
└── system.h / system.cpp  # System facade (Database + DeviceDB + render orchestration)
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `System` | `system.h` | Facade: component registration, transactional mutations with auto GPU sync, undo/redo |

## System API

```cpp
class System {
    // Component registration (host + device)
    template<database::Component T>
    void RegisterComponent(gpu::BufferUsage usage, const std::string& label = "");

    // Transactional mutations with auto GPU sync
    template<typename Fn>
    void Transact(Fn&& fn);

    void Undo();
    void Redo();

    // ECS access
    database::Database& GetDatabase();
    simulate::DeviceDB& GetDeviceDB();

    // GPU buffer access (for render)
    template<database::Component T>
    WGPUBuffer GetBufferHandle() const;
};
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
                    GetBufferHandle<T>()
                              │
                              ▼
                      core_render (reads handles)
```

## Rules

- Namespace: `mps::system`
- System does NOT own RenderEngine — it passes GPU buffer handles to render
- Every Transact/Undo/Redo auto-syncs dirty data to GPU via DeviceDB::Sync()
- core_render has NO dependency on core_database or core_simulate — only reads WGPUBuffer handles
