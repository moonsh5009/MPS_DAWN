#include "core_render/geometry/draw_command.h"
#include "core_render/pass/render_encoder.h"

namespace mps {
namespace render {

void DrawList::Add(DrawCommand cmd) {
    commands_.push_back(std::move(cmd));
}

void DrawList::Clear() {
    commands_.clear();
}

void DrawList::Execute(const RenderEncoder& encoder) const {
    for (const auto& cmd : commands_) {
        for (uint32 i = 0; i < static_cast<uint32>(cmd.vertex_buffers.size()); ++i) {
            encoder.SetVertexBuffer(i, cmd.vertex_buffers[i].buffer, cmd.vertex_buffers[i].offset);
        }
        if (cmd.material_bind_group) {
            encoder.SetBindGroup(1, cmd.material_bind_group);
        }
        if (cmd.index_buffer && cmd.index_count > 0) {
            encoder.SetIndexBuffer(cmd.index_buffer);
            encoder.DrawIndexed(cmd.index_count);
        } else if (cmd.vertex_count > 0) {
            encoder.Draw(cmd.vertex_count);
        }
    }
}

bool DrawList::IsEmpty() const {
    return commands_.empty();
}

uint32 DrawList::GetCount() const {
    return static_cast<uint32>(commands_.size());
}

}  // namespace render
}  // namespace mps
