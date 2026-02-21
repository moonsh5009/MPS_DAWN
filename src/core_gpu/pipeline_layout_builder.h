#pragma once

#include "core_gpu/gpu_types.h"
#include "core_gpu/gpu_handle.h"
#include <vector>
#include <string>

struct WGPUBindGroupLayoutImpl;  typedef WGPUBindGroupLayoutImpl* WGPUBindGroupLayout;

namespace mps {
namespace gpu {

class PipelineLayoutBuilder {
public:
    PipelineLayoutBuilder() = default;
    explicit PipelineLayoutBuilder(const std::string& label);

    PipelineLayoutBuilder(const PipelineLayoutBuilder&) = delete;
    PipelineLayoutBuilder& operator=(const PipelineLayoutBuilder&) = delete;
    PipelineLayoutBuilder(PipelineLayoutBuilder&&) noexcept = default;
    PipelineLayoutBuilder& operator=(PipelineLayoutBuilder&&) noexcept = default;

    PipelineLayoutBuilder&& AddBindGroupLayout(WGPUBindGroupLayout layout) &&;
    GPUPipelineLayout Build() &&;

private:
    std::vector<WGPUBindGroupLayout> layouts_;
    std::string label_;
};

}  // namespace gpu
}  // namespace mps
