#include "ext_mesh/mesh_renderer.h"
#include "ext_mesh/mesh_post_processor.h"
#include "core_simulate/sim_components.h"
#include "core_system/system.h"
#include "core_render/render_engine.h"
#include "core_render/pass/render_encoder.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_layout_builder.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/pipeline_layout_builder.h"
#include "core_render/pipeline/render_pipeline_builder.h"
#include "core_render/uniform/camera_uniform.h"
#include "core_render/uniform/light_uniform.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;
using namespace mps::render;
using namespace mps::simulate;

namespace ext_mesh {

const std::string MeshRenderer::kName = "MeshRenderer";

MeshRenderer::MeshRenderer(system::System& system, MeshPostProcessor& post_processor)
    : system_(system), post_processor_(post_processor) {}

const std::string& MeshRenderer::GetName() const {
    return kName;
}

void MeshRenderer::Initialize(RenderEngine& engine) {
    // Load shaders
    auto vert_shader = ShaderLoader::CreateModule("ext_mesh/mesh_vert.wgsl", "mesh_vert");
    auto frag_shader = ShaderLoader::CreateModule("ext_mesh/mesh_frag.wgsl", "mesh_frag");

    // Bind group layout: camera (binding 0) + light (binding 1)
    bind_group_layout_ = BindGroupLayoutBuilder("mesh_camera_bgl")
        .AddUniformBinding(0, ShaderStage::Vertex | ShaderStage::Fragment)   // camera
        .AddUniformBinding(1, ShaderStage::Fragment)                          // light
        .Build();

    // Create initial bind group
    bind_group_ = BindGroupBuilder("mesh_camera_bg")
        .AddBuffer(0, engine.GetCameraUniform().GetBuffer(), sizeof(CameraUBOData))
        .AddBuffer(1, engine.GetLightUniform().GetBuffer(), sizeof(LightUBOData))
        .Build(bind_group_layout_.GetHandle());

    // Pipeline layout
    auto pipeline_layout = PipelineLayoutBuilder("mesh_layout")
        .AddBindGroupLayout(bind_group_layout_.GetHandle())
        .Build();

    // Render pipeline — TriangleList, double-sided
    pipeline_ = RenderPipelineBuilder("mesh_pipeline")
        .SetPipelineLayout(pipeline_layout.GetHandle())
        .SetVertexShader(vert_shader.GetHandle())
        .SetFragmentShader(frag_shader.GetHandle())
        .AddVertexBufferLayout(VertexStepMode::Vertex, 16, {  // SimPosition (16 bytes)
            {0, VertexFormat::Float32x3, 0},  // position xyz
        })
        .AddVertexBufferLayout(VertexStepMode::Vertex, 16, {  // Normal (16 bytes)
            {1, VertexFormat::Float32x3, 0},  // normal xyz
        })
        .AddColorTarget(engine.GetColorFormat())
        .SetDepthStencil(engine.GetDepthFormat(), true, CompareFunction::Less)
        .SetPrimitive(PrimitiveTopology::TriangleList, CullMode::None, FrontFace::CCW)
        .Build();

    LogInfo("MeshRenderer: pipeline created");
}

void MeshRenderer::Render(RenderEngine& engine, WGPURenderPassEncoder pass) {
    if (!pipeline_.IsValid()) return;

    // Get position buffer from DeviceDB
    WGPUBuffer pos_buf = system_.GetDeviceBuffer<SimPosition>();
    if (!pos_buf) return;

    // Get normal and index buffers from the post-processor
    WGPUBuffer normal_buf = post_processor_.GetNormalBuffer();
    WGPUBuffer index_buf = post_processor_.GetIndexBuffer();
    if (!normal_buf || !index_buf) return;

    uint32 face_count = post_processor_.GetFaceCount();
    if (face_count == 0) return;

    // Recreate bind group if uniform buffers changed (e.g., after resize)
    WGPUBuffer camera_buf = engine.GetCameraUniform().GetBuffer();
    WGPUBuffer light_buf = engine.GetLightUniform().GetBuffer();
    if (camera_buf && light_buf) {
        bind_group_ = BindGroupBuilder("mesh_camera_bg")
            .AddBuffer(0, camera_buf, sizeof(CameraUBOData))
            .AddBuffer(1, light_buf, sizeof(LightUBOData))
            .Build(bind_group_layout_.GetHandle());
    }

    RenderEncoder enc(pass);
    enc.SetPipeline(pipeline_.GetHandle());
    enc.SetBindGroup(0, bind_group_.GetHandle());
    enc.SetVertexBuffer(0, pos_buf);
    enc.SetVertexBuffer(1, normal_buf);
    enc.SetIndexBuffer(index_buf);
    enc.DrawIndexed(face_count * 3);
}

void MeshRenderer::Shutdown() {
    pipeline_ = {};
    bind_group_ = {};
    bind_group_layout_ = {};
    LogInfo("MeshRenderer: shutdown");
}

int32 MeshRenderer::GetOrder() const {
    return 500;
}

}  // namespace ext_mesh
