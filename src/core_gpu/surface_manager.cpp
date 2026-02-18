#include "core_gpu/surface_manager.h"
#include "core_gpu/gpu_core.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <cassert>

using namespace mps::util;

namespace mps {
namespace gpu {

// -- Destruction --------------------------------------------------------------

SurfaceManager::~SurfaceManager() {
    Shutdown();
}

// -- Lifecycle ----------------------------------------------------------------

void SurfaceManager::Initialize(WGPUSurface surface, const SurfaceConfig& config) {
    assert(surface);
    surface_ = surface;
    width_ = config.width;
    height_ = config.height;
    vsync_ = config.vsync;

    // Query surface capabilities for preferred format
    auto& gpu = GPUCore::GetInstance();
    assert(gpu.IsInitialized());

    WGPUSurfaceCapabilities caps = WGPU_SURFACE_CAPABILITIES_INIT;
    wgpuSurfaceGetCapabilities(surface_, gpu.GetAdapter(), &caps);
    if (caps.formatCount > 0) {
        format_ = static_cast<TextureFormat>(caps.formats[0]);
    }
    wgpuSurfaceCapabilitiesFreeMembers(caps);

    Configure();
    initialized_ = true;

    LogInfo("SurfaceManager initialized: ", width_, "x", height_,
            " vsync=", vsync_ ? "on" : "off");
}

void SurfaceManager::Shutdown() {
    if (!initialized_) return;

    ReleaseFrameView();
    // Surface is owned externally (by platform layer); do not release it here
    surface_ = nullptr;
    initialized_ = false;

    LogInfo("SurfaceManager shutdown");
}

void SurfaceManager::Resize(uint32 width, uint32 height) {
    if (width == 0 || height == 0) return;
    if (width == width_ && height == height_) return;

    width_ = width;
    height_ = height;

    ReleaseFrameView();
    Configure();

    LogInfo("SurfaceManager resized: ", width_, "x", height_);
}

// -- Frame acquisition --------------------------------------------------------

WGPUTextureView SurfaceManager::AcquireNextFrameView() {
    assert(initialized_);
    ReleaseFrameView();

    WGPUSurfaceTexture surface_tex = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(surface_, &surface_tex);

    if (surface_tex.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        surface_tex.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        LogError("Failed to acquire surface texture, status: ",
                 static_cast<uint32>(surface_tex.status));
        return nullptr;
    }

    current_view_ = wgpuTextureCreateView(surface_tex.texture, nullptr);
    return current_view_;
}

void SurfaceManager::Present() {
    assert(initialized_);
    wgpuSurfacePresent(surface_);
    ReleaseFrameView();
}

// -- Accessors ----------------------------------------------------------------

TextureFormat SurfaceManager::GetFormat() const { return format_; }
uint32 SurfaceManager::GetWidth() const { return width_; }
uint32 SurfaceManager::GetHeight() const { return height_; }
bool SurfaceManager::IsInitialized() const { return initialized_; }

// -- Internal -----------------------------------------------------------------

void SurfaceManager::Configure() {
    auto& gpu = GPUCore::GetInstance();
    assert(gpu.IsInitialized());

    WGPUSurfaceConfiguration config = WGPU_SURFACE_CONFIGURATION_INIT;
    config.device = gpu.GetDevice();
    config.format = static_cast<WGPUTextureFormat>(format_);
    config.width = width_;
    config.height = height_;
    config.usage = WGPUTextureUsage_RenderAttachment;
    config.presentMode = vsync_ ? WGPUPresentMode_Fifo : WGPUPresentMode_Mailbox;
    config.alphaMode = WGPUCompositeAlphaMode_Auto;
    wgpuSurfaceConfigure(surface_, &config);
}

void SurfaceManager::ReleaseFrameView() {
    if (current_view_) {
        wgpuTextureViewRelease(current_view_);
        current_view_ = nullptr;
    }
}

}  // namespace gpu
}  // namespace mps
