#include "ext_cloth/cloth_renderer.h"
#include "ext_cloth/cloth_components.h"
#include "ext_cloth/cloth_simulator.h"
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
#include "core_simulate/simulator.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;
using namespace mps::render;

namespace ext_cloth {

extern ClothSimulator* g_cloth_simulator;

const std::string ClothRenderer::kName = "ClothRenderer";

ClothRenderer::ClothRenderer(system::System& system)
    : system_(system) {}

const std::string& ClothRenderer::GetName() const {
    return kName;
}

void ClothRenderer::Initialize(RenderEngine& engine) {
    // Load shaders
    auto vert_shader = ShaderLoader::CreateModule("ext_cloth/cloth_vert.wgsl", "cloth_vert");
    auto frag_shader = ShaderLoader::CreateModule("ext_cloth/cloth_frag.wgsl", "cloth_frag");

    // Bind group layout: camera (binding 0) + light (binding 1)
    bind_group_layout_ = BindGroupLayoutBuilder("cloth_camera_bgl")
        .AddUniformBinding(0, ShaderStage::Vertex | ShaderStage::Fragment)   // camera
        .AddUniformBinding(1, ShaderStage::Fragment)                          // light
        .Build();

    // Create initial bind group
    bind_group_ = BindGroupBuilder("cloth_camera_bg")
        .AddBuffer(0, engine.GetCameraUniform().GetBuffer(), sizeof(CameraUBOData))
        .AddBuffer(1, engine.GetLightUniform().GetBuffer(), sizeof(LightUBOData))
        .Build(bind_group_layout_);

    // Pipeline layout
    auto pipeline_layout = PipelineLayoutBuilder("cloth_layout")
        .AddBindGroupLayout(bind_group_layout_)
        .Build();

    // Render pipeline — TriangleList, double-sided
    pipeline_ = RenderPipelineBuilder("cloth_pipeline")
        .SetPipelineLayout(pipeline_layout)
        .SetVertexShader(vert_shader.GetHandle())
        .SetFragmentShader(frag_shader.GetHandle())
        .AddVertexBufferLayout(VertexStepMode::Vertex, 16, {  // ClothPosition (16 bytes)
            {0, VertexFormat::Float32x3, 0},  // position xyz
        })
        .AddVertexBufferLayout(VertexStepMode::Vertex, 16, {  // Normal (16 bytes)
            {1, VertexFormat::Float32x3, 0},  // normal xyz
        })
        .AddColorTarget(engine.GetColorFormat())
        .SetDepthStencil(engine.GetDepthFormat(), true, CompareFunction::Less)
        .SetPrimitive(PrimitiveTopology::TriangleList, CullMode::None, FrontFace::CCW)
        .Build();

    wgpuPipelineLayoutRelease(pipeline_layout);

    LogInfo("ClothRenderer: pipeline created");
}

void ClothRenderer::Render(RenderEngine& engine, WGPURenderPassEncoder pass) {
    if (!pipeline_) {
        LogInfo("ClothRenderer: no pipeline");
        return;
    }

    // Get position buffer from DeviceDB
    WGPUBuffer pos_buf = system_.GetDeviceBuffer<ClothPosition>();
    if (!pos_buf) {
        LogInfo("ClothRenderer: no position buffer");
        return;
    }

    // Find the ClothSimulator to get normal buffer and index buffer
    // We access it through the system's simulator list... but System doesn't expose that directly.
    // The simulator stores normal/index buffer handles we need.
    // For now, get them through a static or callback pattern.
    // Since the simulator is always created before the renderer, and we have a reference to system_,
    // let's just find the simulator.

    // Actually, the cleanest approach: the renderer gets the buffers from the system.
    // But System only exposes GetDeviceBuffer<T> for ECS components.
    // We need to access the simulator's standalone GPU buffers.
    // The simplest approach: cast through the ISimulator* list... but that's not exposed.

    // For now, use a direct approach: search for the cloth simulator using a static pointer.
    // This is a pragmatic solution for an extension — both are in the same module.
    static ClothSimulator* cached_sim = nullptr;
    if (!cached_sim) {
        // The simulator was registered before us; find it by name.
        // Since System doesn't expose the simulator list, we need another approach.
        // Let's use a static registration pattern within the extension.
        // Actually, the simplest: the extension object registered both, so it can wire them.
        // But we don't have that reference here.
        // Pragmatic: use a module-level static that the simulator sets.
        // For now, skip rendering if we can't find the simulator.
        // The GetNormalBuffer/GetIndexBuffer are on ClothSimulator.
        // We'll set up a static pointer approach.
    }

    // Fallback: render without normals (flat shading) using just positions + index
    // For the initial version, we can use a simple index buffer approach.

    // Get the simulator's buffers through a simpler mechanism:
    // The ClothExtension can wire them, or we can use a registry.
    // For simplicity, use a static approach within the extension cpp.

    // Let's just not have a dependency on the simulator and instead look for the normal buffer
    // in a known GPU buffer location. We'll initialize with a default normal buffer.

    // Since both are in ext_cloth and we need this to work, let's use a simple static registry.
    if (!g_cloth_simulator) {
        LogInfo("ClothRenderer: no simulator");
        return;
    }

    WGPUBuffer normal_buf = g_cloth_simulator->GetNormalBuffer();
    WGPUBuffer index_buf = g_cloth_simulator->GetIndexBuffer();
    if (!normal_buf || !index_buf) {
        LogInfo("ClothRenderer: no normal/index buffer (normal=", (normal_buf != nullptr),
                " index=", (index_buf != nullptr), ")");
        return;
    }

    uint32 face_count = g_cloth_simulator->GetFaceCount();
    if (face_count == 0) {
        LogInfo("ClothRenderer: face_count=0");
        return;
    }

    // Recreate bind group if uniform buffers changed (e.g., after resize)
    WGPUBuffer camera_buf = engine.GetCameraUniform().GetBuffer();
    WGPUBuffer light_buf = engine.GetLightUniform().GetBuffer();
    if (camera_buf && light_buf) {
        wgpuBindGroupRelease(bind_group_);
        bind_group_ = BindGroupBuilder("cloth_camera_bg")
            .AddBuffer(0, camera_buf, sizeof(CameraUBOData))
            .AddBuffer(1, light_buf, sizeof(LightUBOData))
            .Build(bind_group_layout_);
    }

    RenderEncoder enc(pass);
    enc.SetPipeline(pipeline_);
    enc.SetBindGroup(0, bind_group_);
    enc.SetVertexBuffer(0, pos_buf);
    enc.SetVertexBuffer(1, normal_buf);
    enc.SetIndexBuffer(index_buf);
    enc.DrawIndexed(face_count * 3);
}

void ClothRenderer::Shutdown() {
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
    LogInfo("ClothRenderer: shutdown");
}

int32 ClothRenderer::GetOrder() const {
    return 500;  // Render before sample points (1000)
}

}  // namespace ext_cloth
