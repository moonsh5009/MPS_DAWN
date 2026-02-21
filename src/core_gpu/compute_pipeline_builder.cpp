#include "core_gpu/compute_pipeline_builder.h"
#include "core_gpu/gpu_core.h"
#include <webgpu/webgpu.h>
#include <utility>

namespace mps {
namespace gpu {

ComputePipelineBuilder::ComputePipelineBuilder(const std::string& label)
    : label_(label) {}

ComputePipelineBuilder&& ComputePipelineBuilder::SetPipelineLayout(
    WGPUPipelineLayout layout) && {
    pipeline_layout_ = layout;
    return std::move(*this);
}

ComputePipelineBuilder&& ComputePipelineBuilder::SetComputeShader(
    WGPUShaderModule module, const std::string& entry) && {
    compute_shader_ = module;
    compute_entry_ = entry;
    return std::move(*this);
}

GPUComputePipeline ComputePipelineBuilder::Build() && {
    auto& gpu = GPUCore::GetInstance();

    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.label = {label_.data(), label_.size()};
    desc.layout = pipeline_layout_;
    desc.compute.module = compute_shader_;
    desc.compute.entryPoint = {compute_entry_.data(), compute_entry_.size()};

    return GPUComputePipeline(wgpuDeviceCreateComputePipeline(gpu.GetDevice(), &desc));
}

}  // namespace gpu
}  // namespace mps
