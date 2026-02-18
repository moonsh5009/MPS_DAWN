#pragma once

#include "core_render/render_types.h"

namespace mps {
namespace render {

class RenderEncoder {
public:
    explicit RenderEncoder(WGPURenderPassEncoder pass);
    explicit RenderEncoder(WGPURenderBundleEncoder bundle);

    void SetPipeline(WGPURenderPipeline pipeline) const;
    void SetBindGroup(uint32 group_index, WGPUBindGroup group,
                      uint32 dynamic_offset_count = 0, const uint32* offsets = nullptr) const;
    void SetVertexBuffer(uint32 slot, WGPUBuffer buffer, uint64 offset = 0, uint64 size = 0) const;
    void SetIndexBuffer(WGPUBuffer buffer, uint64 offset = 0, uint64 size = 0) const;

    void Draw(uint32 vertex_count, uint32 instance_count = 1,
              uint32 first_vertex = 0, uint32 first_instance = 0) const;
    void DrawIndexed(uint32 index_count, uint32 instance_count = 1,
                     uint32 first_index = 0, int32 base_vertex = 0, uint32 first_instance = 0) const;

private:
    WGPURenderPassEncoder pass_ = nullptr;
    WGPURenderBundleEncoder bundle_ = nullptr;
};

}  // namespace render
}  // namespace mps
