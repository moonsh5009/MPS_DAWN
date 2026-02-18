#include "core_render/render_engine.h"
#include "core_gpu/gpu_core.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <cassert>

using namespace mps::util;

namespace mps {
namespace render {

RenderEngine::RenderEngine()
    : camera_controller_(camera_) {}

RenderEngine::~RenderEngine() {
    Shutdown();
}

void RenderEngine::Initialize(WGPUSurface surface, uint32 width, uint32 height,
                               const RenderEngineConfig& config) {
    config_ = config;
    width_ = width;
    height_ = height;

    // Initialize surface
    gpu::SurfaceConfig surface_config;
    surface_config.width = width;
    surface_config.height = height;
    surface_config.vsync = true;
    surface_manager_.Initialize(surface, surface_config);

    // Initialize camera
    camera_.SetAspectRatio(static_cast<float32>(width) / static_cast<float32>(height));

    // Initialize uniforms
    camera_uniform_.Initialize();
    light_uniform_.Initialize();

    // Initialize depth target
    depth_target_ = std::make_unique<RenderTarget>(
        config_.depth_format,
        gpu::TextureUsage::RenderAttachment
    );
    depth_target_->Resize(width, height);

    // Initialize post-processing if enabled
    if (config_.enable_fxaa) {
        fxaa_pass_.Initialize(surface_manager_.GetFormat());
    }
    if (config_.enable_wboit) {
        wboit_pass_.Initialize(surface_manager_.GetFormat());
        wboit_pass_.Resize(width, height);
    }

    initialized_ = true;
    LogInfo("RenderEngine initialized (", width, "x", height, ")");
}

void RenderEngine::Shutdown() {
    if (!initialized_) return;

    if (current_encoder_) {
        wgpuCommandEncoderRelease(current_encoder_);
        current_encoder_ = nullptr;
    }

    depth_target_.reset();
    surface_manager_.Shutdown();
    initialized_ = false;
    LogInfo("RenderEngine shutdown");
}

void RenderEngine::Resize(uint32 width, uint32 height) {
    if (width == 0 || height == 0) return;
    width_ = width;
    height_ = height;

    surface_manager_.Resize(width, height);
    camera_.SetAspectRatio(static_cast<float32>(width) / static_cast<float32>(height));
    depth_target_->Resize(width, height);

    if (config_.enable_wboit) {
        wboit_pass_.Resize(width, height);
    }

    LogInfo("RenderEngine resized (", width, "x", height, ")");
}

void RenderEngine::UpdateUniforms(float32 dt) {
    camera_controller_.Update(dt);
    camera_uniform_.Update(camera_, width_, height_);
    light_uniform_.Update();
}

bool RenderEngine::BeginFrame() {
    assert(initialized_);

    // Acquire surface texture
    current_frame_view_ = surface_manager_.AcquireNextFrameView();
    if (!current_frame_view_) {
        return false;  // Window minimized or surface lost
    }

    // Create command encoder
    auto& gpu_core = gpu::GPUCore::GetInstance();
    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    enc_desc.label = {"render_frame", 12};
    current_encoder_ = wgpuDeviceCreateCommandEncoder(gpu_core.GetDevice(), &enc_desc);

    return true;
}

WGPUCommandEncoder RenderEngine::GetEncoder() const {
    return current_encoder_;
}

WGPUTextureView RenderEngine::GetFrameView() const {
    return current_frame_view_;
}

void RenderEngine::EndFrame() {
    assert(current_encoder_);

    auto& gpu_core = gpu::GPUCore::GetInstance();

    // Finish command encoder and submit
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(current_encoder_, nullptr);
    wgpuQueueSubmit(gpu_core.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(current_encoder_);
    current_encoder_ = nullptr;

    // Present
    surface_manager_.Present();
    current_frame_view_ = nullptr;
}

// Sub-system accessors

gpu::SurfaceManager& RenderEngine::GetSurface() { return surface_manager_; }
Camera& RenderEngine::GetCamera() { return camera_; }
CameraController& RenderEngine::GetCameraController() { return camera_controller_; }
CameraUniform& RenderEngine::GetCameraUniform() { return camera_uniform_; }
LightUniform& RenderEngine::GetLightUniform() { return light_uniform_; }
RenderTarget& RenderEngine::GetDepthTarget() { return *depth_target_; }
FXAAPass& RenderEngine::GetFXAAPass() { return fxaa_pass_; }
WBOITPass& RenderEngine::GetWBOITPass() { return wboit_pass_; }

// Convenience accessors

gpu::TextureFormat RenderEngine::GetColorFormat() const {
    return surface_manager_.GetFormat();
}

gpu::TextureFormat RenderEngine::GetDepthFormat() const {
    return config_.depth_format;
}

uint32 RenderEngine::GetWidth() const {
    return width_;
}

uint32 RenderEngine::GetHeight() const {
    return height_;
}

}  // namespace render
}  // namespace mps
