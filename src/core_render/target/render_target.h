#pragma once

#include "core_render/render_types.h"
#include "core_gpu/gpu_types.h"
#include <memory>

namespace mps {
namespace gpu { class GPUTexture; }
namespace render {

class RenderTarget {
public:
    RenderTarget(gpu::TextureFormat format, gpu::TextureUsage usage);
    ~RenderTarget();

    RenderTarget(RenderTarget&&) noexcept;
    RenderTarget& operator=(RenderTarget&&) noexcept;
    RenderTarget(const RenderTarget&) = delete;
    RenderTarget& operator=(const RenderTarget&) = delete;

    void Resize(uint32 width, uint32 height);
    WGPUTextureView GetView() const;
    gpu::TextureFormat GetFormat() const;
    uint32 GetWidth() const;
    uint32 GetHeight() const;

private:
    gpu::TextureFormat format_;
    gpu::TextureUsage usage_;
    uint32 width_ = 0;
    uint32 height_ = 0;
    std::unique_ptr<gpu::GPUTexture> texture_;
};

}  // namespace render
}  // namespace mps
