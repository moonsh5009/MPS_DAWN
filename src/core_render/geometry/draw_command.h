#pragma once

#include "core_render/render_types.h"
#include <vector>

namespace mps {
namespace render {

class RenderEncoder;

struct VertexBufferBinding {
    WGPUBuffer buffer;
    uint64 stride;
    uint64 offset = 0;
};

struct DrawCommand {
    std::vector<VertexBufferBinding> vertex_buffers;
    WGPUBuffer index_buffer = nullptr;
    uint32 vertex_count = 0;
    uint32 index_count = 0;
    WGPUBindGroup material_bind_group = nullptr;
};

class DrawList {
public:
    void Add(DrawCommand cmd);
    void Clear();
    void Execute(const RenderEncoder& encoder) const;
    bool IsEmpty() const;
    uint32 GetCount() const;

private:
    std::vector<DrawCommand> commands_;
};

}  // namespace render
}  // namespace mps
