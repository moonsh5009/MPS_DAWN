#pragma once

#include "core_gpu/gpu_types.h"
#include <string>

struct WGPUSurfaceImpl;      typedef WGPUSurfaceImpl*     WGPUSurface;
struct WGPUTextureViewImpl;  typedef WGPUTextureViewImpl* WGPUTextureView;

namespace mps {
namespace gpu {

struct SurfaceConfig {
    uint32 width = 1280;
    uint32 height = 720;
    bool vsync = true;
};

class SurfaceManager {
public:
    SurfaceManager() = default;
    ~SurfaceManager();

    SurfaceManager(const SurfaceManager&) = delete;
    SurfaceManager& operator=(const SurfaceManager&) = delete;

    void Initialize(WGPUSurface surface, const SurfaceConfig& config);
    void Shutdown();
    void Resize(uint32 width, uint32 height);

    WGPUTextureView AcquireNextFrameView();
    void Present();

    TextureFormat GetFormat() const;
    uint32 GetWidth() const;
    uint32 GetHeight() const;
    bool IsInitialized() const;

private:
    void Configure();
    void ReleaseFrameView();

    WGPUSurface surface_ = nullptr;
    WGPUTextureView current_view_ = nullptr;
    TextureFormat format_ = TextureFormat::BGRA8Unorm;
    uint32 width_ = 0;
    uint32 height_ = 0;
    bool vsync_ = true;
    bool initialized_ = false;
};

}  // namespace gpu
}  // namespace mps
