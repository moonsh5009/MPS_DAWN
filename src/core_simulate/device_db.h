#pragma once

#include "core_database/component_type.h"
#include "core_database/database.h"
#include "core_gpu/gpu_types.h"
#include "core_simulate/device_buffer_entry.h"
#include <memory>
#include <string>
#include <unordered_map>

// Forward-declare WebGPU handle types
struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace mps {
namespace simulate {

// Mirrors host ECS component data into GPU buffers.
// Register component types, then call Sync() each frame to upload dirty data.
class DeviceDB {
public:
    explicit DeviceDB(database::Database& host_db);

    // Register a component type for GPU mirroring.
    // extra_usage is ORed with the base usage (Storage | CopySrc | CopyDst).
    template<database::Component T>
    void Register(gpu::BufferUsage extra_usage = gpu::BufferUsage::None,
                  const std::string& label = "");

    // Upload all dirty component types to GPU, then clear dirty flags.
    void Sync();

    // Get the GPU buffer handle for a registered component type.
    template<database::Component T>
    WGPUBuffer GetBufferHandle() const;

    // Get a type-erased entry by component type id.
    IDeviceBufferEntry* GetEntryById(database::ComponentTypeId id) const;

    // Check if a component type is registered.
    bool IsRegistered(database::ComponentTypeId id) const;

private:
    database::Database& host_db_;
    std::unordered_map<database::ComponentTypeId, std::unique_ptr<IDeviceBufferEntry>> entries_;
};

// ============================================================================
// Template implementations
// ============================================================================

template<database::Component T>
void DeviceDB::Register(gpu::BufferUsage extra_usage, const std::string& label) {
    database::ComponentTypeId id = database::GetComponentTypeId<T>();
    if (entries_.contains(id)) {
        return;
    }
    entries_.emplace(id, std::make_unique<DeviceBufferEntry<T>>(extra_usage, label));
}

template<database::Component T>
WGPUBuffer DeviceDB::GetBufferHandle() const {
    database::ComponentTypeId id = database::GetComponentTypeId<T>();
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return nullptr;
    }
    return it->second->GetBufferHandle();
}

}  // namespace simulate
}  // namespace mps
