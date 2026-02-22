#pragma once

#include "core_database/component_type.h"
#include "core_database/database.h"
#include "core_gpu/gpu_types.h"
#include "core_simulate/device_db.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Forward-declare WebGPU handle types
struct WGPUBufferImpl;              typedef WGPUBufferImpl* WGPUBuffer;
struct WGPURenderPassEncoderImpl;   typedef WGPURenderPassEncoderImpl* WGPURenderPassEncoder;
struct WGPUSurfaceImpl;             typedef WGPUSurfaceImpl* WGPUSurface;

namespace mps {
namespace platform { class IWindow; }
namespace simulate { class ISimulator; class IDynamicsTermProvider; class IProjectiveTermProvider; }
namespace render { class IObjectRenderer; class RenderEngine; }
namespace system {

class IExtension;

// Top-level system controller.
// Owns the window, GPU lifecycle, render engine, host Database, DeviceDB,
// extension system, and the main loop with simulation controls.
class System {
public:
    System();
    ~System();

    // --- Lifecycle ---
    // Creates window, GPU, and RenderEngine. Call before AddExtension/Transact.
    bool Initialize();

    // Initializes extensions, enters main loop. Call after scene setup.
    // Cleanup happens in ~System().
    void Run();

    // --- Simulation control ---
    bool IsSimulationRunning() const;
    void SetSimulationRunning(bool running);
    void ResetSimulation();

    // --- Component registration ---
    template<database::Component T>
    void RegisterComponent(gpu::BufferUsage extra_usage = gpu::BufferUsage::None,
                           const std::string& label = "");

    // --- Array registration (concatenated GPU buffers from ArrayStorage) ---
    template<database::Component T>
    void RegisterArray(gpu::BufferUsage extra_usage = gpu::BufferUsage::None,
                       const std::string& label = "");

    // --- Indexed array registration (topology arrays with per-entity offset transform) ---
    template<database::Component T, database::Component RefT>
    void RegisterIndexedArray(gpu::BufferUsage extra_usage, const std::string& label,
                              simulate::IndexOffsetFn<T> offset_fn);

    // --- Array queries ---
    template<database::Component T>
    uint32 GetArrayTotalCount() const;

    // Get a type-erased array entry by component type id.
    simulate::IDeviceArrayEntry* GetArrayEntryById(database::ComponentTypeId id) const;

    // --- Transactions ---
    // Execute fn inside a transaction, then sync to GPU (deferred if GPU not ready).
    void Transact(std::function<void(database::Database&)> fn);
    void Undo();
    void Redo();
    bool CanUndo() const;
    bool CanRedo() const;

    // --- GPU buffer access ---
    template<database::Component T>
    WGPUBuffer GetDeviceBuffer() const;

    // --- Read-only queries ---
    template<database::Component T>
    uint32 GetComponentCount() const;

    // Explicit GPU -> Database readback (on-demand, no transaction).
    template<database::Component T>
    void Snapshot();

    // Access the host database.
    const database::Database& GetDatabase() const;
    database::Database& GetDatabase();

    // Access the device DB (GPU mirror).
    simulate::DeviceDB& GetDeviceDB();

    // --- Extension registration (called by extensions during Register) ---
    void AddExtension(std::unique_ptr<IExtension> extension);
    void AddSimulator(std::unique_ptr<simulate::ISimulator> simulator);
    void AddRenderer(std::unique_ptr<render::IObjectRenderer> renderer);

    // --- Term provider registry (for Newton system) ---
    void RegisterTermProvider(database::ComponentTypeId config_type,
                              std::unique_ptr<simulate::IDynamicsTermProvider> provider);

    // Find a provider whose config component exists on the given entity.
    // Returns nullptr if no match.
    simulate::IDynamicsTermProvider* FindTermProvider(database::Entity constraint_entity) const;

    // Find ALL providers whose config component exists on the given entity.
    std::vector<simulate::IDynamicsTermProvider*> FindAllTermProviders(
        database::Entity constraint_entity) const;

    // --- PD term provider registry (for Projective Dynamics system) ---
    void RegisterPDTermProvider(database::ComponentTypeId config_type,
                                std::unique_ptr<simulate::IProjectiveTermProvider> provider);

    simulate::IProjectiveTermProvider* FindPDTermProvider(database::Entity constraint_entity) const;

    std::vector<simulate::IProjectiveTermProvider*> FindAllPDTermProviders(
        database::Entity constraint_entity) const;

private:
    void InitializeExtensions();
    void ShutdownExtensions();
    void UpdateSimulators();
    void RunFrame();
    void RenderFrame();
    void SyncToDevice();
    void NotifyDatabaseChanged();
    void FinishGPUInit();
    std::vector<uint8> ReadbackBuffer(WGPUBuffer src, uint64 size);

#ifdef __EMSCRIPTEN__
    static void EmscriptenMainLoop(void* arg);
#endif

    database::Database db_;
    simulate::DeviceDB device_db_;

    // Window + render engine (created in Run())
    std::unique_ptr<platform::IWindow> window_;
    std::unique_ptr<render::RenderEngine> engine_;

    // Extension system
    std::vector<std::unique_ptr<IExtension>> extensions_;
    std::vector<std::unique_ptr<simulate::ISimulator>> simulators_;
    std::vector<std::unique_ptr<render::IObjectRenderer>> renderers_;
    bool extensions_initialized_ = false;

    // Simulation state
    bool simulation_running_ = false;

    // WASM async GPU initialization
    WGPUSurface pending_surface_ = nullptr;
    bool gpu_ready_ = false;

    // Term provider registry: config component type → provider
    std::unordered_map<database::ComponentTypeId,
                       std::unique_ptr<simulate::IDynamicsTermProvider>> term_providers_;

    // PD term provider registry: config component type → provider
    std::unordered_map<database::ComponentTypeId,
                       std::unique_ptr<simulate::IProjectiveTermProvider>> pd_term_providers_;
};

// ============================================================================
// Template implementations
// ============================================================================

template<database::Component T>
void System::RegisterComponent(gpu::BufferUsage extra_usage, const std::string& label) {
    device_db_.Register<T>(extra_usage, label);
}

template<database::Component T>
void System::RegisterArray(gpu::BufferUsage extra_usage, const std::string& label) {
    device_db_.RegisterArray<T>(extra_usage, label);
}

template<database::Component T, database::Component RefT>
void System::RegisterIndexedArray(gpu::BufferUsage extra_usage, const std::string& label,
                                   simulate::IndexOffsetFn<T> offset_fn) {
    device_db_.RegisterIndexedArray<T, RefT>(extra_usage, label, std::move(offset_fn));
}

template<database::Component T>
WGPUBuffer System::GetDeviceBuffer() const {
    return device_db_.GetBufferHandle<T>();
}

template<database::Component T>
uint32 System::GetArrayTotalCount() const {
    return device_db_.GetArrayTotalCount<T>();
}

template<database::Component T>
uint32 System::GetComponentCount() const {
    auto* storage = db_.GetStorageById(database::GetComponentTypeId<T>());
    if (!storage) return 0;
    return storage->GetDenseCount();
}

template<database::Component T>
void System::Snapshot() {
    WGPUBuffer gpu_buf = device_db_.GetBufferHandle<T>();
    if (!gpu_buf) return;

    auto* storage = db_.GetStorageById(database::GetComponentTypeId<T>());
    if (!storage) return;

    auto* typed = static_cast<database::ComponentStorage<T>*>(storage);
    uint32 count = typed->GetDenseCount();
    if (count == 0) return;

    uint64 byte_size = uint64(count) * sizeof(T);
    auto data = ReadbackBuffer(gpu_buf, byte_size);
    if (data.empty()) return;

    const auto& entities = typed->GetEntities();
    auto* components = reinterpret_cast<const T*>(data.data());
    for (uint32 i = 0; i < count; ++i) {
        db_.DirectSetComponent<T>(entities[i], components[i]);
    }
}

}  // namespace system
}  // namespace mps
