#pragma once

#include "core_render/object_renderer.h"
#include <string>

struct WGPURenderPipelineImpl;  typedef WGPURenderPipelineImpl* WGPURenderPipeline;
struct WGPUBindGroupImpl;       typedef WGPUBindGroupImpl* WGPUBindGroup;
struct WGPUBindGroupLayoutImpl; typedef WGPUBindGroupLayoutImpl* WGPUBindGroupLayout;

namespace mps { namespace system { class System; } }

namespace ext_sample {

class SampleRenderer : public mps::render::IObjectRenderer {
public:
    explicit SampleRenderer(mps::system::System& system);

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize(mps::render::RenderEngine& engine) override;
    void Render(mps::render::RenderEngine& engine, WGPURenderPassEncoder pass) override;
    void Shutdown() override;
    [[nodiscard]] mps::int32 GetOrder() const override;

private:
    mps::system::System& system_;

    WGPURenderPipeline pipeline_ = nullptr;
    WGPUBindGroup bind_group_ = nullptr;
    WGPUBindGroupLayout bind_group_layout_ = nullptr;

    static const std::string kName;
};

}  // namespace ext_sample
