#pragma once

#include "core_render/render_types.h"
#include "core_gpu/surface_manager.h"
#include "core_render/camera/camera.h"
#include "core_render/camera/camera_controller.h"
#include "core_render/uniform/camera_uniform.h"
#include "core_render/uniform/light_uniform.h"
#include "core_render/target/render_target.h"
#include "core_render/post/fxaa_pass.h"
#include "core_render/post/wboit_pass.h"
#include "core_gpu/gpu_types.h"
#include <memory>

namespace mps {
namespace render {

struct RenderEngineConfig {
    ClearColor clear_color = {0.1, 0.1, 0.15, 1.0};
    gpu::TextureFormat depth_format = gpu::TextureFormat::Depth24Plus;
    bool enable_fxaa = false;
    bool enable_wboit = false;
};

class RenderEngine {
public:
    RenderEngine();
    ~RenderEngine();

    RenderEngine(const RenderEngine&) = delete;
    RenderEngine& operator=(const RenderEngine&) = delete;

    void Initialize(WGPUSurface surface, uint32 width, uint32 height,
                    const RenderEngineConfig& config = {});
    void Shutdown();
    void Resize(uint32 width, uint32 height);

    // Uniform updates (camera controller + all uniforms with dirty check)
    void UpdateUniforms(float32 dt);

    // Per-frame cycle
    bool BeginFrame();
    WGPUCommandEncoder GetEncoder() const;
    WGPUTextureView GetFrameView() const;
    void EndFrame();

    // Sub-systems
    gpu::SurfaceManager& GetSurface();
    Camera& GetCamera();
    CameraController& GetCameraController();
    CameraUniform& GetCameraUniform();
    LightUniform& GetLightUniform();
    RenderTarget& GetDepthTarget();

    // Post-processing
    FXAAPass& GetFXAAPass();
    WBOITPass& GetWBOITPass();

    // Convenience
    gpu::TextureFormat GetColorFormat() const;
    gpu::TextureFormat GetDepthFormat() const;
    uint32 GetWidth() const;
    uint32 GetHeight() const;

private:
    gpu::SurfaceManager surface_manager_;
    Camera camera_;
    CameraController camera_controller_;
    CameraUniform camera_uniform_;
    LightUniform light_uniform_;
    std::unique_ptr<RenderTarget> depth_target_;
    FXAAPass fxaa_pass_;
    WBOITPass wboit_pass_;

    RenderEngineConfig config_;
    WGPUCommandEncoder current_encoder_ = nullptr;
    WGPUTextureView current_frame_view_ = nullptr;
    uint32 width_ = 0;
    uint32 height_ = 0;
    bool initialized_ = false;
};

}  // namespace render
}  // namespace mps
