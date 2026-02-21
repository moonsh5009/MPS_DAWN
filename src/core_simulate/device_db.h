#pragma once

#include "core_database/component_type.h"
#include "core_database/database.h"
#include "core_gpu/gpu_types.h"
#include "core_simulate/device_array_buffer.h"
#include "core_simulate/device_buffer_entry.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

// Forward-declare WebGPU handle types
struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace mps {
namespace simulate {

// Mirrors host ECS data (components and arrays) into GPU buffers.
// Register component/array types, then call Sync() each frame to upload dirty data.
class DeviceDB {
public:
    explicit DeviceDB(database::Database& host_db);

    // Register a component type for GPU mirroring (sparse-set backed).
    template<database::Component T>
    void Register(gpu::BufferUsage extra_usage = gpu::BufferUsage::None,
                  const std::string& label = "");

    // Register an array type for GPU mirroring (concatenated from ArrayStorage).
    template<database::Component T>
    void RegisterArray(gpu::BufferUsage extra_usage = gpu::BufferUsage::None,
                       const std::string& label = "");

    // Register an indexed array type that references another array for offset computation.
    // During GPU upload, each element's indices are shifted by the reference array's
    // entity offset so that local per-mesh indices become global buffer indices.
    template<database::Component T, database::Component RefT>
    void RegisterIndexedArray(gpu::BufferUsage extra_usage, const std::string& label,
                              IndexOffsetFn<T> offset_fn);

    // Upload all dirty data to GPU, then clear dirty flags.
    void Sync();

    // Re-upload all registered types to GPU, ignoring dirty flags.
    void ForceSync();

    // Get the GPU buffer handle for a registered type (checks components, arrays, indexed).
    template<database::Component T>
    WGPUBuffer GetBufferHandle() const;

    // Get total element count for a registered array type (checks arrays + indexed).
    template<database::Component T>
    uint32 GetArrayTotalCount() const;

    // Get a type-erased component entry by id.
    IDeviceBufferEntry* GetEntryById(database::ComponentTypeId id) const;

    // Get a type-erased array entry by id (checks arrays + indexed).
    IDeviceArrayEntry* GetArrayEntryById(database::ComponentTypeId id) const;

    // Check if a component type is registered.
    bool IsRegistered(database::ComponentTypeId id) const;

private:
    database::Database& host_db_;
    std::unordered_map<database::ComponentTypeId, std::unique_ptr<IDeviceBufferEntry>> entries_;
    std::unordered_map<database::ComponentTypeId, std::unique_ptr<IDeviceArrayEntry>> array_entries_;

    // Indexed arrays: topology arrays with index offset transform
    std::unordered_map<database::ComponentTypeId, std::unique_ptr<IDeviceArrayEntry>> indexed_entries_;
    std::unordered_map<database::ComponentTypeId, database::ComponentTypeId> indexed_ref_map_;
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
void DeviceDB::RegisterArray(gpu::BufferUsage extra_usage, const std::string& label) {
    database::ComponentTypeId id = database::GetComponentTypeId<T>();
    if (array_entries_.contains(id)) {
        return;
    }
    array_entries_.emplace(id, std::make_unique<DeviceArrayBuffer<T>>(extra_usage, label));
}

template<database::Component T, database::Component RefT>
void DeviceDB::RegisterIndexedArray(gpu::BufferUsage extra_usage, const std::string& label,
                                     IndexOffsetFn<T> offset_fn) {
    auto id = database::GetComponentTypeId<T>();
    auto ref_id = database::GetComponentTypeId<RefT>();
    if (indexed_entries_.contains(id)) return;

    // Find reference array entry (must be registered before indexed array)
    auto ref_it = array_entries_.find(ref_id);
    IDeviceArrayEntry* ref_ptr = (ref_it != array_entries_.end()) ? ref_it->second.get() : nullptr;

    auto buf = std::make_unique<DeviceArrayBuffer<T>>(extra_usage, label);
    buf->SetOffsetSource(ref_ptr, std::move(offset_fn));
    indexed_entries_.emplace(id, std::move(buf));
    indexed_ref_map_.emplace(id, ref_id);
}

template<database::Component T>
WGPUBuffer DeviceDB::GetBufferHandle() const {
    database::ComponentTypeId id = database::GetComponentTypeId<T>();
    // Check component entries first
    auto it = entries_.find(id);
    if (it != entries_.end()) {
        return it->second->GetBufferHandle();
    }
    // Check array entries
    auto ait = array_entries_.find(id);
    if (ait != array_entries_.end()) {
        return ait->second->GetBufferHandle();
    }
    // Check indexed entries
    auto iit = indexed_entries_.find(id);
    if (iit != indexed_entries_.end()) {
        return iit->second->GetBufferHandle();
    }
    return nullptr;
}

template<database::Component T>
uint32 DeviceDB::GetArrayTotalCount() const {
    database::ComponentTypeId id = database::GetComponentTypeId<T>();
    auto it = array_entries_.find(id);
    if (it != array_entries_.end()) return it->second->GetTotalCount();
    auto iit = indexed_entries_.find(id);
    if (iit != indexed_entries_.end()) return iit->second->GetTotalCount();
    return 0;
}

}  // namespace simulate
}  // namespace mps
