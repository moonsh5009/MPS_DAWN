#include "core_gpu/gpu_core.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>

using namespace mps::util;

namespace {

std::string StringViewToString(WGPUStringView sv) {
    if (!sv.data) return "";
    if (sv.length == SIZE_MAX) return std::string(sv.data);
    return std::string(sv.data, sv.length);
}

const char* BackendTypeToString(WGPUBackendType type) {
    switch (type) {
        case WGPUBackendType_Null:      return "Null";
        case WGPUBackendType_WebGPU:    return "WebGPU";
        case WGPUBackendType_D3D11:     return "D3D11";
        case WGPUBackendType_D3D12:     return "D3D12";
        case WGPUBackendType_Metal:     return "Metal";
        case WGPUBackendType_Vulkan:    return "Vulkan";
        case WGPUBackendType_OpenGL:    return "OpenGL";
        case WGPUBackendType_OpenGLES:  return "OpenGLES";
        default:                        return "Unknown";
    }
}

}  // namespace

namespace mps {
namespace gpu {

// -- Callbacks ----------------------------------------------------------------

struct GPUCore::Callbacks {
    static void OnAdapterReceived(
            WGPURequestAdapterStatus status, WGPUAdapter adapter,
            WGPUStringView message, void* userdata1, void* userdata2) {
        auto* core = static_cast<GPUCore*>(userdata1);
        if (status == WGPURequestAdapterStatus_Success) {
            core->adapter_ = adapter;
            core->state_ = GPUState::RequestingDevice;
#ifdef __EMSCRIPTEN__
            core->RequestDevice();
#endif
        } else {
            LogError("Adapter request failed: ", StringViewToString(message));
            core->state_ = GPUState::Error;
        }
    }

    static void OnDeviceReceived(
            WGPURequestDeviceStatus status, WGPUDevice device,
            WGPUStringView message, void* userdata1, void* userdata2) {
        auto* core = static_cast<GPUCore*>(userdata1);
        if (status == WGPURequestDeviceStatus_Success) {
            core->device_ = device;
            core->queue_ = wgpuDeviceGetQueue(device);
            core->state_ = GPUState::Ready;
        } else {
            LogError("Device request failed: ", StringViewToString(message));
            core->state_ = GPUState::Error;
        }
    }

    static void OnDeviceError(WGPUDevice const* device, WGPUErrorType type,
                               WGPUStringView message, void* userdata1, void* userdata2) {
        const char* type_str = "Unknown";
        switch (type) {
            case WGPUErrorType_Validation: type_str = "Validation"; break;
            case WGPUErrorType_OutOfMemory: type_str = "OutOfMemory"; break;
            case WGPUErrorType_Internal: type_str = "Internal"; break;
            default: break;
        }
        LogError("Dawn [", type_str, "]: ", StringViewToString(message));
    }
};

// -- Singleton ----------------------------------------------------------------

GPUCore& GPUCore::GetInstance() {
    static GPUCore instance;
    return instance;
}

GPUCore::~GPUCore() {
    Shutdown();
}

// -- Lifecycle ----------------------------------------------------------------

bool GPUCore::Initialize(const GPUConfig& config, WGPUSurface compatible_surface) {
    if (state_ != GPUState::Uninitialized) {
        LogWarning("GPUCore already initialized or in progress");
        return false;
    }

    config_ = config;
    compatible_surface_ = compatible_surface;

    if (!CreateInstance()) return false;

    state_ = GPUState::RequestingAdapter;
    if (!RequestAdapter(compatible_surface)) return false;

#ifndef __EMSCRIPTEN__
    // Native: adapter callback already fired synchronously via WaitAny
    if (state_ == GPUState::Error) return false;
    if (!RequestDevice()) return false;
    return state_ == GPUState::Ready;
#else
    // WASM: async — adapter callback will chain to RequestDevice via ProcessEvents
    return true;
#endif
}

void GPUCore::Shutdown() {
    ReleaseResources();
}

// -- State --------------------------------------------------------------------

bool GPUCore::IsInitialized() const {
    return state_ == GPUState::Ready;
}

GPUState GPUCore::GetState() const {
    return state_;
}

// -- Accessors ----------------------------------------------------------------

WGPUInstance GPUCore::GetWGPUInstance() const { return instance_; }
WGPUAdapter GPUCore::GetAdapter() const { return adapter_; }
WGPUDevice GPUCore::GetDevice() const { return device_; }
WGPUQueue GPUCore::GetQueue() const { return queue_; }

// -- Info ---------------------------------------------------------------------

std::string GPUCore::GetAdapterName() const {
    if (!adapter_) return "N/A";

    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    wgpuAdapterGetInfo(adapter_, &info);
    std::string name = StringViewToString(info.description);
    wgpuAdapterInfoFreeMembers(info);
    return name;
}

std::string GPUCore::GetBackendType() const {
    if (!adapter_) return "N/A";

    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    wgpuAdapterGetInfo(adapter_, &info);
    std::string backend = BackendTypeToString(info.backendType);
    wgpuAdapterInfoFreeMembers(info);
    return backend;
}

// -- Events -------------------------------------------------------------------

void GPUCore::ProcessEvents() {
    if (instance_) {
        wgpuInstanceProcessEvents(instance_);
    }
}

// -- Surface creation ---------------------------------------------------------

WGPUSurface GPUCore::CreateSurface(void* native_window, void* native_display) {
    // Ensure instance exists
    if (!instance_) {
        if (!CreateInstance()) {
            LogError("Failed to create instance for surface creation");
            return nullptr;
        }
    }

#ifdef __EMSCRIPTEN__
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvasDesc = {};
    canvasDesc.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
    canvasDesc.selector = {"#canvas", 7};

    WGPUSurfaceDescriptor surfaceDesc = {};
    surfaceDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&canvasDesc);
#elif defined(_WIN32)
    WGPUSurfaceSourceWindowsHWND surfaceSource = {};
    surfaceSource.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
    surfaceSource.hinstance = native_display;
    surfaceSource.hwnd = native_window;

    WGPUSurfaceDescriptor surfaceDesc = {};
    surfaceDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&surfaceSource);
#else
    #error "Unsupported platform for surface creation"
#endif

    WGPUSurface surface = wgpuInstanceCreateSurface(instance_, &surfaceDesc);
    if (!surface) {
        LogError("Failed to create WebGPU surface");
        return nullptr;
    }
    LogInfo("WebGPU surface created successfully");
    return surface;
}

// -- Internal -----------------------------------------------------------------

bool GPUCore::CreateInstance() {
    if (instance_) return true;  // Already created (e.g., by CreateSurface)

    WGPUInstanceDescriptor desc = WGPU_INSTANCE_DESCRIPTOR_INIT;

#ifndef __EMSCRIPTEN__
    // Enable timed wait for synchronous WaitAny calls (native only)
    WGPUInstanceFeatureName features[] = { WGPUInstanceFeatureName_TimedWaitAny };
    desc.requiredFeatureCount = 1;
    desc.requiredFeatures = features;
#endif

    instance_ = wgpuCreateInstance(&desc);

    if (!instance_) {
        LogError("Failed to create WGPUInstance");
        state_ = GPUState::Error;
        return false;
    }

    LogInfo("WGPUInstance created");
    return true;
}

bool GPUCore::RequestAdapter(WGPUSurface compatible_surface) {
    WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    options.powerPreference = config_.prefer_high_performance
        ? WGPUPowerPreference_HighPerformance
        : WGPUPowerPreference_LowPower;
    options.compatibleSurface = compatible_surface;

    WGPURequestAdapterCallbackInfo cb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
#ifdef __EMSCRIPTEN__
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
#else
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
#endif
    cb.callback = Callbacks::OnAdapterReceived;
    cb.userdata1 = this;

    WGPUFuture future = wgpuInstanceRequestAdapter(instance_, &options, cb);

#ifndef __EMSCRIPTEN__
    // Synchronous wait (native)
    WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
    wait.future = future;
    wgpuInstanceWaitAny(instance_, 1, &wait, UINT64_MAX);

    if (!adapter_) {
        LogError("Adapter request did not complete successfully");
        state_ = GPUState::Error;
        return false;
    }
#endif

    return true;
}

bool GPUCore::RequestDevice() {
    WGPUDeviceDescriptor desc = WGPU_DEVICE_DESCRIPTOR_INIT;
    desc.uncapturedErrorCallbackInfo.callback = Callbacks::OnDeviceError;
    desc.uncapturedErrorCallbackInfo.userdata1 = this;

    WGPURequestDeviceCallbackInfo cb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
#ifdef __EMSCRIPTEN__
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
#else
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
#endif
    cb.callback = Callbacks::OnDeviceReceived;
    cb.userdata1 = this;

    WGPUFuture future = wgpuAdapterRequestDevice(adapter_, &desc, cb);

#ifndef __EMSCRIPTEN__
    // Synchronous wait (native)
    WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
    wait.future = future;
    wgpuInstanceWaitAny(instance_, 1, &wait, UINT64_MAX);

    if (!device_) {
        LogError("Device request did not complete successfully");
        state_ = GPUState::Error;
        return false;
    }
#endif

    return true;
}

void GPUCore::ReleaseResources() {
    if (queue_)    { wgpuQueueRelease(queue_);       queue_ = nullptr; }
    if (device_)   { wgpuDeviceRelease(device_);     device_ = nullptr; }
    if (adapter_)  { wgpuAdapterRelease(adapter_);   adapter_ = nullptr; }
    if (instance_) { wgpuInstanceRelease(instance_);  instance_ = nullptr; }
    state_ = GPUState::Uninitialized;
}

}  // namespace gpu
}  // namespace mps
