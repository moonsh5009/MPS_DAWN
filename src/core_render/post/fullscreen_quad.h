#pragma once

#include "core_render/render_types.h"

namespace mps {
namespace render {

class FullscreenQuad {
public:
    static void Draw(WGPURenderPassEncoder pass);
};

}  // namespace render
}  // namespace mps
