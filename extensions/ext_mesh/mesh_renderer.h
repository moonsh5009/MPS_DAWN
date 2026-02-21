#pragma once

#include "core_render/object_renderer.h"
#include "core_gpu/gpu_handle.h"
#include <string>

namespace mps { namespace system { class System; } }

namespace ext_mesh {

class MeshPostProcessor;

class MeshRenderer : public mps::render::IObjectRenderer {
public:
    MeshRenderer(mps::system::System& system, MeshPostProcessor& post_processor);

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize(mps::render::RenderEngine& engine) override;
    void Render(mps::render::RenderEngine& engine, WGPURenderPassEncoder pass) override;
    void Shutdown() override;
    [[nodiscard]] mps::int32 GetOrder() const override;

private:
    mps::system::System& system_;
    MeshPostProcessor& post_processor_;

    mps::gpu::GPURenderPipeline pipeline_;
    mps::gpu::GPUBindGroup bind_group_;
    mps::gpu::GPUBindGroupLayout bind_group_layout_;

    static const std::string kName;
};

}  // namespace ext_mesh
