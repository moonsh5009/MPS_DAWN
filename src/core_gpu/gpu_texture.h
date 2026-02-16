#pragma once

#include "core_gpu/gpu_types.h"
#include <string>

// Forward-declare WebGPU handle types
struct WGPUTextureImpl;      typedef WGPUTextureImpl*     WGPUTexture;
struct WGPUTextureViewImpl;  typedef WGPUTextureViewImpl* WGPUTextureView;

namespace mps {
namespace gpu {

struct TextureConfig {
    uint32 width = 1;
    uint32 height = 1;
    uint32 depth_or_array_layers = 1;
    TextureFormat format = TextureFormat::RGBA8Unorm;
    TextureUsage usage = TextureUsage::TextureBinding;
    TextureDimension dimension = TextureDimension::D2;
    uint32 mip_level_count = 1;
    uint32 sample_count = 1;
    std::string label;
};

class GPUTexture {
public:
    explicit GPUTexture(const TextureConfig& config);  // throws GPUException
    ~GPUTexture();

    // Move-only
    GPUTexture(GPUTexture&& other) noexcept;
    GPUTexture& operator=(GPUTexture&& other) noexcept;
    GPUTexture(const GPUTexture&) = delete;
    GPUTexture& operator=(const GPUTexture&) = delete;

    void WriteData(const void* data, uint64 data_size, uint32 mip_level = 0);

    WGPUTexture GetHandle() const;
    WGPUTextureView GetView() const;
    TextureFormat GetFormat() const;
    uint32 GetWidth() const;
    uint32 GetHeight() const;

private:
    void Release();

    WGPUTexture handle_ = nullptr;
    WGPUTextureView default_view_ = nullptr;
    TextureConfig config_;
};

}  // namespace gpu
}  // namespace mps
