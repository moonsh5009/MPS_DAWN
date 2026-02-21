#pragma once

#include "core_render/render_types.h"
#include "core_gpu/gpu_types.h"
#include "core_gpu/gpu_handle.h"
#include <memory>

namespace mps {
namespace gpu { class GPUTexture; class GPUSampler; }
namespace render {

class WBOITPass {
public:
    WBOITPass();
    ~WBOITPass();

    WBOITPass(WBOITPass&&) noexcept;
    WBOITPass& operator=(WBOITPass&&) noexcept;
    WBOITPass(const WBOITPass&) = delete;
    WBOITPass& operator=(const WBOITPass&) = delete;

    void Initialize(gpu::TextureFormat output_format);
    void Resize(uint32 width, uint32 height);

    WGPUTextureView GetAccumView() const;
    WGPUTextureView GetRevealView() const;

    void ResetTargets(WGPUCommandEncoder encoder);
    void Compose(WGPUCommandEncoder encoder, WGPUTextureView output_view);

private:
    std::unique_ptr<gpu::GPUTexture> accum_texture_;   // RGBA16Float
    std::unique_ptr<gpu::GPUTexture> reveal_texture_;   // R8Unorm
    gpu::GPURenderPipeline compose_pipeline_;
    gpu::GPUBindGroupLayout compose_bgl_;
    std::unique_ptr<gpu::GPUSampler> sampler_;
    gpu::TextureFormat output_format_ = gpu::TextureFormat::Undefined;
    uint32 width_ = 0;
    uint32 height_ = 0;
    bool initialized_ = false;
};

}  // namespace render
}  // namespace mps
