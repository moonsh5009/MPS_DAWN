#include "core_render/post/wboit_pass.h"
#include "core_gpu/bind_group_layout_builder.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/pipeline_layout_builder.h"
#include "core_render/pipeline/render_pipeline_builder.h"
#include "core_gpu/shader_loader.h"
#include "core_render/post/fullscreen_quad.h"
#include "core_gpu/gpu_texture.h"
#include "core_gpu/gpu_sampler.h"
#include "core_gpu/gpu_core.h"
#include <webgpu/webgpu.h>

namespace mps {
namespace render {

WBOITPass::WBOITPass() = default;
WBOITPass::~WBOITPass() {
    if (compose_pipeline_) wgpuRenderPipelineRelease(compose_pipeline_);
    if (compose_bgl_) wgpuBindGroupLayoutRelease(compose_bgl_);
}

WBOITPass::WBOITPass(WBOITPass&&) noexcept = default;
WBOITPass& WBOITPass::operator=(WBOITPass&&) noexcept = default;

void WBOITPass::Initialize(gpu::TextureFormat output_format) {
    output_format_ = output_format;

    auto vert_shader = gpu::ShaderLoader::CreateModule("post/fullscreen_vert.wgsl", "wboit_compose_vert");
    auto frag_shader = gpu::ShaderLoader::CreateModule("post/wboit_compose_frag.wgsl", "wboit_compose_frag");

    compose_bgl_ = gpu::BindGroupLayoutBuilder("wboit_compose_bgl")
        .AddTextureBinding(0, gpu::ShaderStage::Fragment)    // accum texture
        .AddTextureBinding(1, gpu::ShaderStage::Fragment)    // reveal texture
        .AddSamplerBinding(2, gpu::ShaderStage::Fragment)    // sampler
        .Build();

    auto pipeline_layout = gpu::PipelineLayoutBuilder("wboit_compose_layout")
        .AddBindGroupLayout(compose_bgl_)
        .Build();

    gpu::SamplerConfig sampler_config;
    sampler_config.mag_filter = gpu::FilterMode::Linear;
    sampler_config.min_filter = gpu::FilterMode::Linear;
    sampler_config.label = "wboit_sampler";
    sampler_ = std::make_unique<gpu::GPUSampler>(sampler_config);

    compose_pipeline_ = RenderPipelineBuilder("wboit_compose_pipeline")
        .SetPipelineLayout(pipeline_layout)
        .SetVertexShader(vert_shader.GetHandle())
        .SetFragmentShader(frag_shader.GetHandle())
        .AddColorTarget(output_format)
        .SetPrimitive(gpu::PrimitiveTopology::TriangleList, CullMode::None, FrontFace::CCW)
        .Build();

    wgpuPipelineLayoutRelease(pipeline_layout);
    initialized_ = true;
}

void WBOITPass::Resize(uint32 width, uint32 height) {
    if (width == width_ && height == height_ && accum_texture_) return;
    width_ = width;
    height_ = height;

    // Accumulation texture: RGBA16Float, RenderAttachment + TextureBinding
    gpu::TextureConfig accum_config;
    accum_config.width = width;
    accum_config.height = height;
    accum_config.format = gpu::TextureFormat::RGBA16Float;
    accum_config.usage = gpu::TextureUsage::RenderAttachment | gpu::TextureUsage::TextureBinding;
    accum_config.label = "wboit_accum";
    accum_texture_ = std::make_unique<gpu::GPUTexture>(accum_config);

    // Reveal texture: R8Unorm, RenderAttachment + TextureBinding
    gpu::TextureConfig reveal_config;
    reveal_config.width = width;
    reveal_config.height = height;
    reveal_config.format = gpu::TextureFormat::R8Unorm;
    reveal_config.usage = gpu::TextureUsage::RenderAttachment | gpu::TextureUsage::TextureBinding;
    reveal_config.label = "wboit_reveal";
    reveal_texture_ = std::make_unique<gpu::GPUTexture>(reveal_config);
}

WGPUTextureView WBOITPass::GetAccumView() const {
    return accum_texture_ ? accum_texture_->GetView() : nullptr;
}

WGPUTextureView WBOITPass::GetRevealView() const {
    return reveal_texture_ ? reveal_texture_->GetView() : nullptr;
}

void WBOITPass::ResetTargets(WGPUCommandEncoder encoder) {
    if (!accum_texture_ || !reveal_texture_) return;

    // Clear accum to (0,0,0,0) and reveal to (1,0,0,0)
    WGPURenderPassColorAttachment color_atts[2];

    // Accum attachment: clear to zero
    color_atts[0] = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    color_atts[0].view = accum_texture_->GetView();
    color_atts[0].loadOp = WGPULoadOp_Clear;
    color_atts[0].storeOp = WGPUStoreOp_Store;
    color_atts[0].clearValue = {0.0, 0.0, 0.0, 0.0};

    // Reveal attachment: clear to 1.0 (fully transparent initially)
    color_atts[1] = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    color_atts[1].view = reveal_texture_->GetView();
    color_atts[1].loadOp = WGPULoadOp_Clear;
    color_atts[1].storeOp = WGPUStoreOp_Store;
    color_atts[1].clearValue = {1.0, 0.0, 0.0, 0.0};

    WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    desc.label = {"wboit_reset", 11};
    desc.colorAttachmentCount = 2;
    desc.colorAttachments = color_atts;

    // Empty render pass (just clear, no draw)
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &desc);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
}

void WBOITPass::Compose(WGPUCommandEncoder encoder, WGPUTextureView output_view) {
    if (!initialized_ || !accum_texture_ || !reveal_texture_) return;

    // Create bind group with accum view, reveal view, and sampler
    WGPUBindGroup bind_group = gpu::BindGroupBuilder("wboit_compose_bg")
        .AddTextureView(0, accum_texture_->GetView())
        .AddTextureView(1, reveal_texture_->GetView())
        .AddSampler(2, sampler_->GetHandle())
        .Build(compose_bgl_);

    // Render pass with output view (load=Load to preserve opaque content, store=Store)
    WGPURenderPassColorAttachment color_att = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    color_att.view = output_view;
    color_att.loadOp = WGPULoadOp_Load;
    color_att.storeOp = WGPUStoreOp_Store;

    WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    desc.label = {"wboit_compose", 13};
    desc.colorAttachmentCount = 1;
    desc.colorAttachments = &color_att;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &desc);
    wgpuRenderPassEncoderSetPipeline(pass, compose_pipeline_);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
    FullscreenQuad::Draw(pass);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    wgpuBindGroupRelease(bind_group);
}

}  // namespace render
}  // namespace mps
