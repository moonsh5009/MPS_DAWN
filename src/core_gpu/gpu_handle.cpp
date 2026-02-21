#include "core_gpu/gpu_handle.h"
#include <webgpu/webgpu.h>

namespace mps {
namespace gpu {

template<> void GPUHandle<ComputePipelineTag>::Release() {
    if (handle_) { wgpuComputePipelineRelease(handle_); handle_ = nullptr; }
}

template<> void GPUHandle<RenderPipelineTag>::Release() {
    if (handle_) { wgpuRenderPipelineRelease(handle_); handle_ = nullptr; }
}

template<> void GPUHandle<BindGroupTag>::Release() {
    if (handle_) { wgpuBindGroupRelease(handle_); handle_ = nullptr; }
}

template<> void GPUHandle<BindGroupLayoutTag>::Release() {
    if (handle_) { wgpuBindGroupLayoutRelease(handle_); handle_ = nullptr; }
}

template<> void GPUHandle<PipelineLayoutTag>::Release() {
    if (handle_) { wgpuPipelineLayoutRelease(handle_); handle_ = nullptr; }
}

}  // namespace gpu
}  // namespace mps
