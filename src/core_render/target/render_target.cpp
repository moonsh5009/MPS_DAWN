#include "core_render/target/render_target.h"
#include "core_gpu/gpu_texture.h"

namespace mps {
namespace render {

RenderTarget::RenderTarget(gpu::TextureFormat format, gpu::TextureUsage usage)
    : format_(format), usage_(usage) {}

RenderTarget::~RenderTarget() = default;

RenderTarget::RenderTarget(RenderTarget&&) noexcept = default;
RenderTarget& RenderTarget::operator=(RenderTarget&&) noexcept = default;

void RenderTarget::Resize(uint32 width, uint32 height) {
    if (width == width_ && height == height_ && texture_) return;
    width_ = width;
    height_ = height;

    gpu::TextureConfig config;
    config.width = width;
    config.height = height;
    config.format = format_;
    config.usage = usage_;
    config.label = "render_target";
    texture_ = std::make_unique<gpu::GPUTexture>(config);
}

WGPUTextureView RenderTarget::GetView() const {
    return texture_ ? texture_->GetView() : nullptr;
}

gpu::TextureFormat RenderTarget::GetFormat() const {
    return format_;
}

uint32 RenderTarget::GetWidth() const {
    return width_;
}

uint32 RenderTarget::GetHeight() const {
    return height_;
}

}  // namespace render
}  // namespace mps
