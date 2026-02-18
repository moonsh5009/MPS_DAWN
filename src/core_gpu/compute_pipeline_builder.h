#pragma once

#include "core_gpu/gpu_types.h"
#include <string>

struct WGPUPipelineLayoutImpl;   typedef WGPUPipelineLayoutImpl*   WGPUPipelineLayout;
struct WGPUShaderModuleImpl;     typedef WGPUShaderModuleImpl*     WGPUShaderModule;
struct WGPUComputePipelineImpl;  typedef WGPUComputePipelineImpl*  WGPUComputePipeline;

namespace mps {
namespace gpu {

class ComputePipelineBuilder {
public:
    ComputePipelineBuilder() = default;
    explicit ComputePipelineBuilder(const std::string& label);

    ComputePipelineBuilder(const ComputePipelineBuilder&) = delete;
    ComputePipelineBuilder& operator=(const ComputePipelineBuilder&) = delete;
    ComputePipelineBuilder(ComputePipelineBuilder&&) noexcept = default;
    ComputePipelineBuilder& operator=(ComputePipelineBuilder&&) noexcept = default;

    ComputePipelineBuilder&& SetPipelineLayout(WGPUPipelineLayout layout) &&;
    ComputePipelineBuilder&& SetComputeShader(WGPUShaderModule module,
                                               const std::string& entry = "cs_main") &&;

    WGPUComputePipeline Build() &&;

private:
    std::string label_;
    WGPUPipelineLayout pipeline_layout_ = nullptr;
    WGPUShaderModule compute_shader_ = nullptr;
    std::string compute_entry_ = "cs_main";
};

}  // namespace gpu
}  // namespace mps
