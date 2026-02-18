#pragma once

#include "core_util/types.h"
#include <string>

struct WGPURenderPassEncoderImpl;
typedef WGPURenderPassEncoderImpl* WGPURenderPassEncoder;

namespace mps {
namespace render {

class RenderEngine;

class IObjectRenderer {
public:
    virtual ~IObjectRenderer() = default;

    [[nodiscard]] virtual const std::string& GetName() const = 0;
    virtual void Initialize(RenderEngine& engine) {}
    virtual void Resize(uint32 width, uint32 height) {}
    virtual void Render(RenderEngine& engine, WGPURenderPassEncoder pass) = 0;
    virtual void Shutdown() {}
    [[nodiscard]] virtual int32 GetOrder() const { return 1000; }
};

}  // namespace render
}  // namespace mps
