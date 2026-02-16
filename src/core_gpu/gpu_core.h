#pragma once

#include "core_util/types.h"
#include <string>

// Forward-declare WebGPU handle types (avoid including webgpu.h in headers)
struct WGPUInstanceImpl;   typedef WGPUInstanceImpl*  WGPUInstance;
struct WGPUAdapterImpl;    typedef WGPUAdapterImpl*   WGPUAdapter;
struct WGPUDeviceImpl;     typedef WGPUDeviceImpl*    WGPUDevice;
struct WGPUQueueImpl;      typedef WGPUQueueImpl*     WGPUQueue;
struct WGPUSurfaceImpl;    typedef WGPUSurfaceImpl*   WGPUSurface;

namespace mps {
namespace gpu {

struct GPUConfig {
    bool enable_validation = true;        // Dawn validation (native only)
    bool prefer_high_performance = true;  // WGPUPowerPreference_HighPerformance
};

enum class GPUState : uint8 {
    Uninitialized,
    CreatingInstance,
    RequestingAdapter,
    RequestingDevice,
    Ready,
    Error
};

class GPUCore {
public:
    static GPUCore& GetInstance();

    // Lifecycle
    bool Initialize(const GPUConfig& config = {},
                    WGPUSurface compatible_surface = nullptr);
    void Shutdown();

    // State
    bool IsInitialized() const;
    GPUState GetState() const;

    // Accessors (valid after IsInitialized() == true)
    WGPUInstance GetWGPUInstance() const;
    WGPUAdapter GetAdapter() const;
    WGPUDevice GetDevice() const;
    WGPUQueue GetQueue() const;

    // Info
    std::string GetAdapterName() const;
    std::string GetBackendType() const;

    // Process async events (call in main loop, required for WASM init)
    void ProcessEvents();

private:
    GPUCore() = default;
    ~GPUCore();

    GPUCore(const GPUCore&) = delete;
    GPUCore& operator=(const GPUCore&) = delete;
    GPUCore(GPUCore&&) = delete;
    GPUCore& operator=(GPUCore&&) = delete;

    bool CreateInstance();
    bool RequestAdapter(WGPUSurface compatible_surface);
    bool RequestDevice();
    void ReleaseResources();

    // Callback access — defined in gpu_core.cpp
    struct Callbacks;
    friend struct Callbacks;

    WGPUInstance instance_ = nullptr;
    WGPUAdapter adapter_ = nullptr;
    WGPUDevice device_ = nullptr;
    WGPUQueue queue_ = nullptr;

    GPUState state_ = GPUState::Uninitialized;
    GPUConfig config_;
    WGPUSurface compatible_surface_ = nullptr;  // stored for WASM async flow
};

}  // namespace gpu
}  // namespace mps
