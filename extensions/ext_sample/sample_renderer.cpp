#include "ext_sample/sample_renderer.h"
#include "ext_sample/sample_components.h"
#include "core_system/system.h"
#include "core_render/render_engine.h"
#include "core_render/pass/render_encoder.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_layout_builder.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/pipeline_layout_builder.h"
#include "core_render/pipeline/render_pipeline_builder.h"
#include "core_render/uniform/camera_uniform.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;
using namespace mps::render;

namespace ext_sample {

const std::string SampleRenderer::kName = "SampleRenderer";

SampleRenderer::SampleRenderer(system::System& system)
    : system_(system) {}

const std::string& SampleRenderer::GetName() const {
    return kName;
}

void SampleRenderer::Initialize(RenderEngine& engine) {
    // Load shaders
    auto vert_shader = ShaderLoader::CreateModule("ext_sample/point_vert.wgsl", "sample_point_vert");
    auto frag_shader = ShaderLoader::CreateModule("ext_sample/point_frag.wgsl", "sample_point_frag");

    // Camera bind group layout (group 0, binding 0)
    bind_group_layout_ = BindGroupLayoutBuilder("sample_camera_bgl")
        .AddUniformBinding(0, ShaderStage::Vertex)
        .Build();

    // Camera bind group
    bind_group_ = BindGroupBuilder("sample_camera_bg")
        .AddBuffer(0, engine.GetCameraUniform().GetBuffer(), sizeof(CameraUBOData))
        .Build(bind_group_layout_);

    // Pipeline layout
    auto pipeline_layout = PipelineLayoutBuilder("sample_layout")
        .AddBindGroupLayout(bind_group_layout_)
        .Build();

    // Render pipeline — PointList topology, no index buffer
    pipeline_ = RenderPipelineBuilder("sample_point_pipeline")
        .SetPipelineLayout(pipeline_layout)
        .SetVertexShader(vert_shader.GetHandle())
        .SetFragmentShader(frag_shader.GetHandle())
        .AddVertexBufferLayout(VertexStepMode::Vertex, sizeof(SampleTransform), {
            {0, VertexFormat::Float32x3, 0},  // position (x, y, z)
        })
        .AddColorTarget(engine.GetColorFormat())
        .SetDepthStencil(engine.GetDepthFormat(), true, CompareFunction::Less)
        .SetPrimitive(PrimitiveTopology::PointList, CullMode::None, FrontFace::CCW)
        .Build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    LogInfo("SampleRenderer: pipeline created");
}

void SampleRenderer::Render(RenderEngine& engine, WGPURenderPassEncoder pass) {
    if (!pipeline_) return;

    WGPUBuffer buffer = system_.GetDeviceBuffer<SampleTransform>();
    if (!buffer) return;

    // Recreate bind group if camera buffer changed (e.g. after resize)
    WGPUBuffer camera_buf = engine.GetCameraUniform().GetBuffer();
    if (camera_buf) {
        wgpuBindGroupRelease(bind_group_);
        bind_group_ = BindGroupBuilder("sample_camera_bg")
            .AddBuffer(0, camera_buf, sizeof(CameraUBOData))
            .Build(bind_group_layout_);
    }

    // Get entity count from the database storage
    auto* storage = system_.GetDeviceDB().GetEntryById(
        database::GetComponentTypeId<SampleTransform>());
    if (!storage) return;

    auto& db = system_.GetDatabase();
    auto* component_storage = db.GetStorageById(
        database::GetComponentTypeId<SampleTransform>());
    if (!component_storage) return;

    uint32 count = component_storage->GetDenseCount();
    if (count == 0) return;

    RenderEncoder enc(pass);
    enc.SetPipeline(pipeline_);
    enc.SetBindGroup(0, bind_group_);
    enc.SetVertexBuffer(0, buffer);
    enc.Draw(count);
}

void SampleRenderer::Shutdown() {
    if (pipeline_) {
        wgpuRenderPipelineRelease(pipeline_);
        pipeline_ = nullptr;
    }
    if (bind_group_) {
        wgpuBindGroupRelease(bind_group_);
        bind_group_ = nullptr;
    }
    if (bind_group_layout_) {
        wgpuBindGroupLayoutRelease(bind_group_layout_);
        bind_group_layout_ = nullptr;
    }
    LogInfo("SampleRenderer: shutdown");
}

int32 SampleRenderer::GetOrder() const {
    return 1000;
}

}  // namespace ext_sample
