#pragma once

#include "core_gpu/gpu_types.h"
#include "core_gpu/gpu_handle.h"
#include <vector>
#include <string>

// Forward declarations
struct WGPUBufferImpl;           typedef WGPUBufferImpl*          WGPUBuffer;
struct WGPUTextureViewImpl;      typedef WGPUTextureViewImpl*     WGPUTextureView;
struct WGPUSamplerImpl;          typedef WGPUSamplerImpl*         WGPUSampler;
struct WGPUBindGroupLayoutImpl;  typedef WGPUBindGroupLayoutImpl* WGPUBindGroupLayout;

namespace mps {
namespace gpu {

class BindGroupBuilder {
public:
    BindGroupBuilder() = default;
    explicit BindGroupBuilder(const std::string& label);

    BindGroupBuilder(const BindGroupBuilder&) = delete;
    BindGroupBuilder& operator=(const BindGroupBuilder&) = delete;
    BindGroupBuilder(BindGroupBuilder&&) noexcept = default;
    BindGroupBuilder& operator=(BindGroupBuilder&& other) noexcept;

    BindGroupBuilder&& AddBuffer(uint32 binding, WGPUBuffer buffer, uint64 size, uint64 offset = 0) &&;
    BindGroupBuilder&& AddTextureView(uint32 binding, WGPUTextureView view) &&;
    BindGroupBuilder&& AddSampler(uint32 binding, WGPUSampler sampler) &&;

    GPUBindGroup Build(WGPUBindGroupLayout layout) &&;

private:
    struct Entry {
        uint32 binding;
        WGPUBuffer buffer = nullptr;
        uint64 buffer_size = 0;
        uint64 buffer_offset = 0;
        WGPUTextureView texture_view = nullptr;
        WGPUSampler sampler = nullptr;
    };
    std::vector<Entry> entries_;
    std::string label_;
};

}  // namespace gpu
}  // namespace mps
