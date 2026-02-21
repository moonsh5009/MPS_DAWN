#pragma once

#include <utility>

// Forward declarations (same pattern as existing builders — no webgpu.h in headers)
struct WGPUComputePipelineImpl;  typedef WGPUComputePipelineImpl*  WGPUComputePipeline;
struct WGPURenderPipelineImpl;   typedef WGPURenderPipelineImpl*   WGPURenderPipeline;
struct WGPUBindGroupImpl;        typedef WGPUBindGroupImpl*        WGPUBindGroup;
struct WGPUBindGroupLayoutImpl;  typedef WGPUBindGroupLayoutImpl*  WGPUBindGroupLayout;
struct WGPUPipelineLayoutImpl;   typedef WGPUPipelineLayoutImpl*   WGPUPipelineLayout;

namespace mps {
namespace gpu {

// Tag types for GPUHandle specialization
struct ComputePipelineTag  { using HandleType = WGPUComputePipeline; };
struct RenderPipelineTag   { using HandleType = WGPURenderPipeline; };
struct BindGroupTag        { using HandleType = WGPUBindGroup; };
struct BindGroupLayoutTag  { using HandleType = WGPUBindGroupLayout; };
struct PipelineLayoutTag   { using HandleType = WGPUPipelineLayout; };

// Move-only RAII wrapper for WebGPU handles.
// Destructor calls the appropriate wgpuXxxRelease via explicit specialization.
template<typename Tag>
class GPUHandle {
public:
    using HandleType = typename Tag::HandleType;

    GPUHandle() = default;
    explicit GPUHandle(HandleType handle) : handle_(handle) {}

    ~GPUHandle() { Release(); }

    // Move-only
    GPUHandle(GPUHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    GPUHandle& operator=(GPUHandle&& other) noexcept {
        if (this != &other) {
            Release();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    GPUHandle(const GPUHandle&) = delete;
    GPUHandle& operator=(const GPUHandle&) = delete;

    [[nodiscard]] HandleType GetHandle() const { return handle_; }
    [[nodiscard]] bool IsValid() const { return handle_ != nullptr; }
    explicit operator bool() const { return IsValid(); }

    // Release ownership without calling wgpuXxxRelease
    HandleType Detach() {
        auto h = handle_;
        handle_ = nullptr;
        return h;
    }

private:
    void Release();  // Specialized per tag in gpu_handle.cpp

    HandleType handle_ = nullptr;
};

// Type aliases
using GPUComputePipeline  = GPUHandle<ComputePipelineTag>;
using GPURenderPipeline   = GPUHandle<RenderPipelineTag>;
using GPUBindGroup        = GPUHandle<BindGroupTag>;
using GPUBindGroupLayout  = GPUHandle<BindGroupLayoutTag>;
using GPUPipelineLayout   = GPUHandle<PipelineLayoutTag>;

}  // namespace gpu
}  // namespace mps
