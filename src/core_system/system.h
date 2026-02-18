#pragma once

#include "core_database/component_type.h"
#include "core_database/database.h"
#include "core_gpu/gpu_types.h"
#include "core_simulate/device_db.h"
#include <functional>
#include <string>

// Forward-declare WebGPU handle types
struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace mps {
namespace system {

// Top-level system controller.
// Owns the host Database and DeviceDB, providing a unified facade for
// component registration, transactional mutations with automatic GPU sync,
// and undo/redo.
class System {
public:
    System();
    ~System();

    // Register a component type for both host ECS and GPU mirroring.
    template<database::Component T>
    void RegisterComponent(gpu::BufferUsage extra_usage = gpu::BufferUsage::None,
                           const std::string& label = "");

    // Execute fn inside a transaction, then sync to GPU.
    // If fn throws, the transaction is rolled back and SyncToDevice is NOT called.
    void Transact(std::function<void(database::Database&)> fn);

    // Undo the last transaction and sync to GPU.
    void Undo();

    // Redo the last undone transaction and sync to GPU.
    void Redo();

    bool CanUndo() const;
    bool CanRedo() const;

    // Get the GPU buffer handle for a registered component type.
    template<database::Component T>
    WGPUBuffer GetDeviceBuffer() const;

    // Read-only access to the host database.
    const database::Database& GetDatabase() const;

private:
    void SyncToDevice();

    database::Database db_;
    simulate::DeviceDB device_db_;
};

// ============================================================================
// Template implementations
// ============================================================================

template<database::Component T>
void System::RegisterComponent(gpu::BufferUsage extra_usage, const std::string& label) {
    device_db_.Register<T>(extra_usage, label);
}

template<database::Component T>
WGPUBuffer System::GetDeviceBuffer() const {
    return device_db_.GetBufferHandle<T>();
}

}  // namespace system
}  // namespace mps
