#include "core_render/pass/render_encoder.h"
#include <webgpu/webgpu.h>

namespace mps {
namespace render {

RenderEncoder::RenderEncoder(WGPURenderPassEncoder pass)
    : pass_(pass) {}

RenderEncoder::RenderEncoder(WGPURenderBundleEncoder bundle)
    : bundle_(bundle) {}

void RenderEncoder::SetPipeline(WGPURenderPipeline pipeline) const {
    if (pass_) wgpuRenderPassEncoderSetPipeline(pass_, pipeline);
    else if (bundle_) wgpuRenderBundleEncoderSetPipeline(bundle_, pipeline);
}

void RenderEncoder::SetBindGroup(uint32 group_index, WGPUBindGroup group,
                                  uint32 dynamic_offset_count, const uint32* offsets) const {
    if (pass_) wgpuRenderPassEncoderSetBindGroup(pass_, group_index, group, dynamic_offset_count, offsets);
    else if (bundle_) wgpuRenderBundleEncoderSetBindGroup(bundle_, group_index, group, dynamic_offset_count, offsets);
}

void RenderEncoder::SetVertexBuffer(uint32 slot, WGPUBuffer buffer, uint64 offset, uint64 size) const {
    uint64 actual_size = size > 0 ? size : WGPU_WHOLE_SIZE;
    if (pass_) wgpuRenderPassEncoderSetVertexBuffer(pass_, slot, buffer, offset, actual_size);
    else if (bundle_) wgpuRenderBundleEncoderSetVertexBuffer(bundle_, slot, buffer, offset, actual_size);
}

void RenderEncoder::SetIndexBuffer(WGPUBuffer buffer, uint64 offset, uint64 size) const {
    uint64 actual_size = size > 0 ? size : WGPU_WHOLE_SIZE;
    if (pass_) wgpuRenderPassEncoderSetIndexBuffer(pass_, buffer, WGPUIndexFormat_Uint32, offset, actual_size);
    else if (bundle_) wgpuRenderBundleEncoderSetIndexBuffer(bundle_, buffer, WGPUIndexFormat_Uint32, offset, actual_size);
}

void RenderEncoder::Draw(uint32 vertex_count, uint32 instance_count,
                          uint32 first_vertex, uint32 first_instance) const {
    if (pass_) wgpuRenderPassEncoderDraw(pass_, vertex_count, instance_count, first_vertex, first_instance);
    else if (bundle_) wgpuRenderBundleEncoderDraw(bundle_, vertex_count, instance_count, first_vertex, first_instance);
}

void RenderEncoder::DrawIndexed(uint32 index_count, uint32 instance_count,
                                 uint32 first_index, int32 base_vertex, uint32 first_instance) const {
    if (pass_) wgpuRenderPassEncoderDrawIndexed(pass_, index_count, instance_count, first_index, base_vertex, first_instance);
    else if (bundle_) wgpuRenderBundleEncoderDrawIndexed(bundle_, index_count, instance_count, first_index, base_vertex, first_instance);
}

}  // namespace render
}  // namespace mps
