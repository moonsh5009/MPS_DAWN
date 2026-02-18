#include "core_render/post/fullscreen_quad.h"
#include <webgpu/webgpu.h>

namespace mps {
namespace render {

void FullscreenQuad::Draw(WGPURenderPassEncoder pass) {
    // Draw 3 vertices with no buffer bound.
    // The fullscreen triangle vertex shader generates positions from vertex_index.
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
}

}  // namespace render
}  // namespace mps
