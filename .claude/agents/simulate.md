---
name: simulate
description: Device DB (GPU buffer mirrors) and ISimulator interface. Owns core_simulate module. Use when implementing or modifying simulation logic or GPU buffer sync.
model: opus
memory: project
---

# Simulate Agent

Owns the `core_simulate` module. Manages GPU buffer mirroring of host ECS data (DeviceDB).

## Module Structure

```
src/core_simulate/
├── CMakeLists.txt           # STATIC library → mps::core_simulate (depends: core_util, core_gpu, core_database)
├── device_buffer_entry.h    # IDeviceBufferEntry, DeviceBufferEntry<T> (type-erased GPU buffer wrapper)
├── device_db.h / device_db.cpp  # DeviceDB (GPU mirrors of host ECS data)
└── simulator.h              # ISimulator interface (for extensions)
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `IDeviceBufferEntry` | `device_buffer_entry.h` | Type-erased base for GPU buffer entries |
| `DeviceBufferEntry<T>` | `device_buffer_entry.h` | Owns `gpu::GPUBuffer<T>`, syncs from `IComponentStorage` |
| `DeviceDB` | `device_db.h` | Registers component types, syncs dirty data to GPU |
| `ISimulator` | `simulator.h` | Extension interface: `Update(Database&, dt)`, managed by System |

## DeviceDB API

```cpp
explicit DeviceDB(database::Database& host_db);

template<database::Component T>
void Register(gpu::BufferUsage extra_usage = gpu::BufferUsage::None,
              const std::string& label = "");

void Sync();  // upload dirty types, then ClearAllDirty()

template<database::Component T> WGPUBuffer GetBufferHandle() const;
IDeviceBufferEntry* GetEntryById(database::ComponentTypeId id) const;
bool IsRegistered(database::ComponentTypeId id) const;
```

## Design Patterns

- **Sync strategy**: Full re-upload of dirty dense arrays via `GPUBuffer<T>::WriteData()`
- **Buffer usage**: Base `Storage | CopySrc | CopyDst`, plus extra flags from `Register()`
- **Lazy buffer creation**: Buffer created/resized on first `SyncFromHost()` call
- **Dirty tracking**: Uses `Database::GetDirtyTypeIds()` + `ClearAllDirty()` after sync

## Data Flow

```
Database (host) ──dirty flags──► DeviceDB::Sync() ──WriteData──► GPU Buffers
                                                                    │
core_system calls Sync()                              core_render reads handles
after every Transact/Undo/Redo
```

## Rules

- Namespace: `mps::simulate`
- DeviceDB does NOT own the Database — takes a reference
- core_render reads GPU buffer handles but has NO dependency on core_simulate
- core_system bridges the gap: calls Sync() and passes handles to render
