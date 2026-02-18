#include "core_gpu/compute_encoder.h"
#include <webgpu/webgpu.h>

namespace mps {
namespace gpu {

ComputeEncoder::ComputeEncoder(WGPUComputePassEncoder pass)
    : pass_(pass) {}

void ComputeEncoder::SetPipeline(WGPUComputePipeline pipeline) const {
    wgpuComputePassEncoderSetPipeline(pass_, pipeline);
}

void ComputeEncoder::SetBindGroup(uint32 group_index, WGPUBindGroup group,
                                    uint32 dynamic_offset_count, const uint32* offsets) const {
    wgpuComputePassEncoderSetBindGroup(pass_, group_index, group, dynamic_offset_count, offsets);
}

void ComputeEncoder::Dispatch(uint32 x, uint32 y, uint32 z) const {
    wgpuComputePassEncoderDispatchWorkgroups(pass_, x, y, z);
}

void ComputeEncoder::DispatchIndirect(WGPUBuffer buffer, uint64 offset) const {
    wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass_, buffer, offset);
}

}  // namespace gpu
}  // namespace mps
