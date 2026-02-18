#pragma once

#include "core_render/render_types.h"
#include "core_gpu/gpu_types.h"
#include <vector>
#include <string>
#include <optional>

namespace mps {
namespace render {

class RenderPipelineBuilder {
public:
    RenderPipelineBuilder() = default;
    explicit RenderPipelineBuilder(const std::string& label);

    RenderPipelineBuilder(const RenderPipelineBuilder&) = delete;
    RenderPipelineBuilder& operator=(const RenderPipelineBuilder&) = delete;
    RenderPipelineBuilder(RenderPipelineBuilder&&) noexcept = default;
    RenderPipelineBuilder& operator=(RenderPipelineBuilder&&) noexcept = default;

    RenderPipelineBuilder&& SetPipelineLayout(WGPUPipelineLayout layout) &&;
    RenderPipelineBuilder&& SetVertexShader(WGPUShaderModule module, const std::string& entry = "vs_main") &&;
    RenderPipelineBuilder&& SetFragmentShader(WGPUShaderModule module, const std::string& entry = "fs_main") &&;

    struct VertexAttribute {
        uint32 location;
        gpu::VertexFormat format;
        uint64 offset;
    };
    RenderPipelineBuilder&& AddVertexBufferLayout(gpu::VertexStepMode step_mode, uint64 stride,
                                                   std::vector<VertexAttribute> attributes) &&;

    RenderPipelineBuilder&& AddColorTarget(gpu::TextureFormat format,
                                            std::optional<BlendState> blend = std::nullopt) &&;
    RenderPipelineBuilder&& SetDepthStencil(gpu::TextureFormat format, bool depth_write_enabled,
                                             gpu::CompareFunction compare) &&;
    RenderPipelineBuilder&& SetPrimitive(gpu::PrimitiveTopology topology = gpu::PrimitiveTopology::TriangleList,
                                          CullMode cull = CullMode::Back,
                                          FrontFace front = FrontFace::CCW) &&;

    WGPURenderPipeline Build() &&;

private:
    std::string label_;
    WGPUPipelineLayout pipeline_layout_ = nullptr;
    WGPUShaderModule vertex_shader_ = nullptr;
    std::string vertex_entry_ = "vs_main";
    WGPUShaderModule fragment_shader_ = nullptr;
    std::string fragment_entry_ = "fs_main";

    struct VertexBufferLayoutData {
        gpu::VertexStepMode step_mode;
        uint64 stride;
        std::vector<VertexAttribute> attributes;
    };
    std::vector<VertexBufferLayoutData> vertex_buffer_layouts_;

    struct ColorTargetData {
        gpu::TextureFormat format;
        std::optional<BlendState> blend;
    };
    std::vector<ColorTargetData> color_targets_;

    struct DepthStencilData {
        gpu::TextureFormat format;
        bool depth_write_enabled;
        gpu::CompareFunction compare;
    };
    std::optional<DepthStencilData> depth_stencil_;

    gpu::PrimitiveTopology topology_ = gpu::PrimitiveTopology::TriangleList;
    CullMode cull_mode_ = CullMode::Back;
    FrontFace front_face_ = FrontFace::CCW;
};

}  // namespace render
}  // namespace mps
