#include "core_gpu/gpu_texture.h"
#include "core_gpu/gpu_core.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <cassert>
#include <utility>

using namespace mps::util;

namespace mps {
namespace gpu {

// -- Construction -------------------------------------------------------------

GPUTexture::GPUTexture(const TextureConfig& config)
    : config_(config) {
    auto& core = GPUCore::GetInstance();
    assert(core.IsInitialized());

    WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    desc.label = {config.label.data(), config.label.size()};
    desc.usage = static_cast<WGPUTextureUsage>(config.usage);
    desc.dimension = static_cast<WGPUTextureDimension>(config.dimension);
    desc.size = {config.width, config.height, config.depth_or_array_layers};
    desc.format = static_cast<WGPUTextureFormat>(config.format);
    desc.mipLevelCount = config.mip_level_count;
    desc.sampleCount = config.sample_count;

    handle_ = wgpuDeviceCreateTexture(core.GetDevice(), &desc);
    if (!handle_) {
        throw GPUException("Failed to create GPU texture: " + config.label);
    }

    // Create default view
    default_view_ = wgpuTextureCreateView(handle_, nullptr);
    if (!default_view_) {
        wgpuTextureRelease(handle_);
        handle_ = nullptr;
        throw GPUException("Failed to create default texture view: " + config.label);
    }

    LogInfo("GPUTexture created: ", config.label,
            " (", config.width, "x", config.height, ")");
}

GPUTexture::~GPUTexture() {
    Release();
}

// -- Move semantics -----------------------------------------------------------

GPUTexture::GPUTexture(GPUTexture&& other) noexcept
    : handle_(other.handle_), default_view_(other.default_view_), config_(std::move(other.config_)) {
    other.handle_ = nullptr;
    other.default_view_ = nullptr;
}

GPUTexture& GPUTexture::operator=(GPUTexture&& other) noexcept {
    if (this != &other) {
        Release();
        handle_ = other.handle_;
        default_view_ = other.default_view_;
        config_ = std::move(other.config_);
        other.handle_ = nullptr;
        other.default_view_ = nullptr;
    }
    return *this;
}

// -- Data operations ----------------------------------------------------------

void GPUTexture::WriteData(const void* data, uint64 data_size, uint32 mip_level) {
    assert(handle_);
    auto& core = GPUCore::GetInstance();

    // Compute bytes per row based on format
    uint32 bytes_per_pixel = 4;  // Default for RGBA8
    switch (config_.format) {
        case TextureFormat::R8Unorm:
        case TextureFormat::R8Snorm:
        case TextureFormat::R8Uint:
        case TextureFormat::R8Sint:
            bytes_per_pixel = 1;
            break;
        case TextureFormat::RG8Unorm:
        case TextureFormat::RG8Snorm:
        case TextureFormat::R16Float:
            bytes_per_pixel = 2;
            break;
        case TextureFormat::RGBA8Unorm:
        case TextureFormat::RGBA8UnormSrgb:
        case TextureFormat::RGBA8Snorm:
        case TextureFormat::RGBA8Uint:
        case TextureFormat::RGBA8Sint:
        case TextureFormat::BGRA8Unorm:
        case TextureFormat::BGRA8UnormSrgb:
        case TextureFormat::RGB10A2Unorm:
        case TextureFormat::R32Float:
        case TextureFormat::R32Uint:
        case TextureFormat::R32Sint:
            bytes_per_pixel = 4;
            break;
        case TextureFormat::RG32Float:
        case TextureFormat::RG16Float:
        case TextureFormat::RGBA16Float:
            bytes_per_pixel = 8;
            break;
        case TextureFormat::RGBA32Float:
            bytes_per_pixel = 16;
            break;
        default:
            bytes_per_pixel = 4;
            break;
    }

    uint32 mip_width = config_.width >> mip_level;
    uint32 mip_height = config_.height >> mip_level;
    if (mip_width == 0) mip_width = 1;
    if (mip_height == 0) mip_height = 1;

    WGPUTexelCopyTextureInfo dest = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    dest.texture = handle_;
    dest.mipLevel = mip_level;

    WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    layout.offset = 0;
    layout.bytesPerRow = mip_width * bytes_per_pixel;
    layout.rowsPerImage = mip_height;

    WGPUExtent3D write_size = {mip_width, mip_height, config_.depth_or_array_layers};

    wgpuQueueWriteTexture(core.GetQueue(), &dest, data,
                           static_cast<size_t>(data_size), &layout, &write_size);
}

// -- Accessors ----------------------------------------------------------------

WGPUTexture GPUTexture::GetHandle() const { return handle_; }
WGPUTextureView GPUTexture::GetView() const { return default_view_; }
TextureFormat GPUTexture::GetFormat() const { return config_.format; }
uint32 GPUTexture::GetWidth() const { return config_.width; }
uint32 GPUTexture::GetHeight() const { return config_.height; }

// -- Internal -----------------------------------------------------------------

void GPUTexture::Release() {
    if (default_view_) {
        wgpuTextureViewRelease(default_view_);
        default_view_ = nullptr;
    }
    if (handle_) {
        wgpuTextureRelease(handle_);
        handle_ = nullptr;
    }
}

}  // namespace gpu
}  // namespace mps
