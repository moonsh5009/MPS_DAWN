#pragma once

#include "core_render/render_types.h"
#include "core_gpu/gpu_types.h"
#include "core_gpu/gpu_handle.h"
#include <memory>

namespace mps {
namespace gpu { class GPUShader; class GPUSampler; }
namespace render {

class FXAAPass {
public:
    FXAAPass();
    ~FXAAPass();

    FXAAPass(FXAAPass&&) noexcept;
    FXAAPass& operator=(FXAAPass&&) noexcept;
    FXAAPass(const FXAAPass&) = delete;
    FXAAPass& operator=(const FXAAPass&) = delete;

    void Initialize(gpu::TextureFormat output_format);
    void Execute(WGPUCommandEncoder encoder, WGPUTextureView input_view,
                 WGPUTextureView output_view, uint32 width, uint32 height);

private:
    gpu::GPURenderPipeline pipeline_;
    gpu::GPUBindGroupLayout bind_group_layout_;
    std::unique_ptr<gpu::GPUSampler> sampler_;
    bool initialized_ = false;
};

}  // namespace render
}  // namespace mps
