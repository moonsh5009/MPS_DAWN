#include "core_render/pipeline/render_pipeline_builder.h"
#include "core_gpu/gpu_core.h"
#include <webgpu/webgpu.h>
#include <utility>

namespace mps {
namespace render {

RenderPipelineBuilder::RenderPipelineBuilder(const std::string& label)
    : label_(label) {}

RenderPipelineBuilder&& RenderPipelineBuilder::SetPipelineLayout(
    WGPUPipelineLayout layout) && {
    pipeline_layout_ = layout;
    return std::move(*this);
}

RenderPipelineBuilder&& RenderPipelineBuilder::SetVertexShader(
    WGPUShaderModule module, const std::string& entry) && {
    vertex_shader_ = module;
    vertex_entry_ = entry;
    return std::move(*this);
}

RenderPipelineBuilder&& RenderPipelineBuilder::SetFragmentShader(
    WGPUShaderModule module, const std::string& entry) && {
    fragment_shader_ = module;
    fragment_entry_ = entry;
    return std::move(*this);
}

RenderPipelineBuilder&& RenderPipelineBuilder::AddVertexBufferLayout(
    gpu::VertexStepMode step_mode, uint64 stride,
    std::vector<VertexAttribute> attributes) && {
    vertex_buffer_layouts_.push_back({step_mode, stride, std::move(attributes)});
    return std::move(*this);
}

RenderPipelineBuilder&& RenderPipelineBuilder::AddColorTarget(
    gpu::TextureFormat format, std::optional<BlendState> blend) && {
    color_targets_.push_back({format, blend});
    return std::move(*this);
}

RenderPipelineBuilder&& RenderPipelineBuilder::SetDepthStencil(
    gpu::TextureFormat format, bool depth_write_enabled,
    gpu::CompareFunction compare) && {
    depth_stencil_ = DepthStencilData{format, depth_write_enabled, compare};
    return std::move(*this);
}

RenderPipelineBuilder&& RenderPipelineBuilder::SetPrimitive(
    gpu::PrimitiveTopology topology, CullMode cull, FrontFace front) && {
    topology_ = topology;
    cull_mode_ = cull;
    front_face_ = front;
    return std::move(*this);
}

WGPURenderPipeline RenderPipelineBuilder::Build() && {
    auto& gpu = gpu::GPUCore::GetInstance();

    // -- Vertex attributes and buffer layouts --
    // Keep attribute vectors alive until pipeline creation
    std::vector<std::vector<WGPUVertexAttribute>> all_attributes;
    std::vector<WGPUVertexBufferLayout> vb_layouts;

    for (const auto& vbl : vertex_buffer_layouts_) {
        auto& attrs = all_attributes.emplace_back();
        for (const auto& a : vbl.attributes) {
            WGPUVertexAttribute attr = WGPU_VERTEX_ATTRIBUTE_INIT;
            attr.format = static_cast<WGPUVertexFormat>(a.format);
            attr.offset = a.offset;
            attr.shaderLocation = a.location;
            attrs.push_back(attr);
        }
        WGPUVertexBufferLayout layout = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
        layout.arrayStride = vbl.stride;
        layout.stepMode = static_cast<WGPUVertexStepMode>(vbl.step_mode);
        layout.attributeCount = attrs.size();
        layout.attributes = attrs.data();
        vb_layouts.push_back(layout);
    }

    // -- Vertex state --
    WGPUVertexState vertex_state = WGPU_VERTEX_STATE_INIT;
    vertex_state.module = vertex_shader_;
    vertex_state.entryPoint = {vertex_entry_.data(), vertex_entry_.size()};
    vertex_state.bufferCount = vb_layouts.size();
    vertex_state.buffers = vb_layouts.data();

    // -- Color targets --
    std::vector<WGPUColorTargetState> color_target_states;
    std::vector<WGPUBlendState> blend_states;  // keep alive

    for (const auto& ct : color_targets_) {
        WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
        target.format = static_cast<WGPUTextureFormat>(ct.format);
        target.writeMask = WGPUColorWriteMask_All;

        if (ct.blend) {
            WGPUBlendState blend = WGPU_BLEND_STATE_INIT;
            blend.color.srcFactor = static_cast<WGPUBlendFactor>(ct.blend->src_color);
            blend.color.dstFactor = static_cast<WGPUBlendFactor>(ct.blend->dst_color);
            blend.color.operation = static_cast<WGPUBlendOperation>(ct.blend->color_op);
            blend.alpha.srcFactor = static_cast<WGPUBlendFactor>(ct.blend->src_alpha);
            blend.alpha.dstFactor = static_cast<WGPUBlendFactor>(ct.blend->dst_alpha);
            blend.alpha.operation = static_cast<WGPUBlendOperation>(ct.blend->alpha_op);
            blend_states.push_back(blend);
            target.blend = &blend_states.back();
        }
        color_target_states.push_back(target);
    }

    // -- Fragment state --
    WGPUFragmentState fragment_state = WGPU_FRAGMENT_STATE_INIT;
    fragment_state.module = fragment_shader_;
    fragment_state.entryPoint = {fragment_entry_.data(), fragment_entry_.size()};
    fragment_state.targetCount = color_target_states.size();
    fragment_state.targets = color_target_states.data();

    // -- Depth stencil --
    WGPUDepthStencilState depth_stencil_state = WGPU_DEPTH_STENCIL_STATE_INIT;
    if (depth_stencil_) {
        depth_stencil_state.format = static_cast<WGPUTextureFormat>(depth_stencil_->format);
        depth_stencil_state.depthWriteEnabled = depth_stencil_->depth_write_enabled
            ? WGPUOptionalBool_True : WGPUOptionalBool_False;
        depth_stencil_state.depthCompare = static_cast<WGPUCompareFunction>(
            depth_stencil_->compare);
    }

    // -- Primitive state --
    WGPUPrimitiveState primitive_state = WGPU_PRIMITIVE_STATE_INIT;
    primitive_state.topology = static_cast<WGPUPrimitiveTopology>(topology_);
    primitive_state.frontFace = static_cast<WGPUFrontFace>(front_face_);
    primitive_state.cullMode = static_cast<WGPUCullMode>(cull_mode_);

    // -- Pipeline descriptor --
    WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    desc.label = {label_.data(), label_.size()};
    desc.layout = pipeline_layout_;
    desc.vertex = vertex_state;
    desc.primitive = primitive_state;
    desc.multisample.count = 1;
    desc.multisample.mask = 0xFFFFFFFF;

    if (fragment_shader_) {
        desc.fragment = &fragment_state;
    }
    if (depth_stencil_) {
        desc.depthStencil = &depth_stencil_state;
    }

    return wgpuDeviceCreateRenderPipeline(gpu.GetDevice(), &desc);
}

}  // namespace render
}  // namespace mps
