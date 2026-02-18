#include "core_gpu/pipeline_layout_builder.h"
#include "core_gpu/gpu_core.h"
#include <webgpu/webgpu.h>
#include <utility>

namespace mps {
namespace gpu {

PipelineLayoutBuilder::PipelineLayoutBuilder(const std::string& label)
    : label_(label) {}

PipelineLayoutBuilder&& PipelineLayoutBuilder::AddBindGroupLayout(
    WGPUBindGroupLayout layout) && {
    layouts_.push_back(layout);
    return std::move(*this);
}

WGPUPipelineLayout PipelineLayoutBuilder::Build() && {
    auto& gpu = GPUCore::GetInstance();

    WGPUPipelineLayoutDescriptor desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    desc.label = {label_.data(), label_.size()};
    desc.bindGroupLayoutCount = layouts_.size();
    desc.bindGroupLayouts = layouts_.data();

    return wgpuDeviceCreatePipelineLayout(gpu.GetDevice(), &desc);
}

}  // namespace gpu
}  // namespace mps
