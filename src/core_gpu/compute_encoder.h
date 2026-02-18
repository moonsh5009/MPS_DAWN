#pragma once

#include "core_util/types.h"

struct WGPUComputePassEncoderImpl;  typedef WGPUComputePassEncoderImpl* WGPUComputePassEncoder;
struct WGPUComputePipelineImpl;     typedef WGPUComputePipelineImpl*    WGPUComputePipeline;
struct WGPUBindGroupImpl;           typedef WGPUBindGroupImpl*          WGPUBindGroup;
struct WGPUBufferImpl;              typedef WGPUBufferImpl*             WGPUBuffer;

namespace mps {
namespace gpu {

class ComputeEncoder {
public:
    explicit ComputeEncoder(WGPUComputePassEncoder pass);

    void SetPipeline(WGPUComputePipeline pipeline) const;
    void SetBindGroup(uint32 group_index, WGPUBindGroup group,
                      uint32 dynamic_offset_count = 0, const uint32* offsets = nullptr) const;
    void Dispatch(uint32 x, uint32 y = 1, uint32 z = 1) const;
    void DispatchIndirect(WGPUBuffer buffer, uint64 offset = 0) const;

private:
    WGPUComputePassEncoder pass_;
};

}  // namespace gpu
}  // namespace mps
