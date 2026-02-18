#include "core_render/post/fxaa_pass.h"
#include "core_gpu/bind_group_layout_builder.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/pipeline_layout_builder.h"
#include "core_render/pipeline/render_pipeline_builder.h"
#include "core_gpu/shader_loader.h"
#include "core_render/post/fullscreen_quad.h"
#include "core_gpu/gpu_sampler.h"
#include "core_gpu/gpu_core.h"
#include <webgpu/webgpu.h>

namespace mps {
namespace render {

FXAAPass::FXAAPass() = default;
FXAAPass::~FXAAPass() {
    if (pipeline_) wgpuRenderPipelineRelease(pipeline_);
    if (bind_group_layout_) wgpuBindGroupLayoutRelease(bind_group_layout_);
}

FXAAPass::FXAAPass(FXAAPass&&) noexcept = default;
FXAAPass& FXAAPass::operator=(FXAAPass&&) noexcept = default;

void FXAAPass::Initialize(gpu::TextureFormat output_format) {
    auto vert_shader = gpu::ShaderLoader::CreateModule("post/fullscreen_vert.wgsl", "fxaa_vert");
    auto frag_shader = gpu::ShaderLoader::CreateModule("post/fxaa_frag.wgsl", "fxaa_frag");

    bind_group_layout_ = gpu::BindGroupLayoutBuilder("fxaa_bgl")
        .AddTextureBinding(0, gpu::ShaderStage::Fragment)
        .AddSamplerBinding(1, gpu::ShaderStage::Fragment)
        .Build();

    auto pipeline_layout = gpu::PipelineLayoutBuilder("fxaa_layout")
        .AddBindGroupLayout(bind_group_layout_)
        .Build();

    gpu::SamplerConfig sampler_config;
    sampler_config.mag_filter = gpu::FilterMode::Linear;
    sampler_config.min_filter = gpu::FilterMode::Linear;
    sampler_config.label = "fxaa_sampler";
    sampler_ = std::make_unique<gpu::GPUSampler>(sampler_config);

    pipeline_ = RenderPipelineBuilder("fxaa_pipeline")
        .SetPipelineLayout(pipeline_layout)
        .SetVertexShader(vert_shader.GetHandle())
        .SetFragmentShader(frag_shader.GetHandle())
        .AddColorTarget(output_format)
        .SetPrimitive(gpu::PrimitiveTopology::TriangleList, CullMode::None, FrontFace::CCW)
        .Build();

    wgpuPipelineLayoutRelease(pipeline_layout);
    initialized_ = true;
}

void FXAAPass::Execute(WGPUCommandEncoder encoder, WGPUTextureView input_view,
                        WGPUTextureView output_view, uint32 width, uint32 height) {
    if (!initialized_) return;

    // Create bind group with input texture and sampler
    WGPUBindGroup bind_group = gpu::BindGroupBuilder("fxaa_bg")
        .AddTextureView(0, input_view)
        .AddSampler(1, sampler_->GetHandle())
        .Build(bind_group_layout_);

    // Begin render pass targeting the output view
    WGPURenderPassColorAttachment color_att = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    color_att.view = output_view;
    color_att.loadOp = WGPULoadOp_Undefined;
    color_att.storeOp = WGPUStoreOp_Store;

    WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    desc.label = {"fxaa_pass", 9};
    desc.colorAttachmentCount = 1;
    desc.colorAttachments = &color_att;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &desc);
    wgpuRenderPassEncoderSetPipeline(pass, pipeline_);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
    FullscreenQuad::Draw(pass);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    wgpuBindGroupRelease(bind_group);
}

}  // namespace render
}  // namespace mps
