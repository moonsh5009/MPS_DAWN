#pragma once

#include "core_render/render_types.h"
#include <vector>
#include <functional>
#include <string>

namespace mps {
namespace render {

class RenderPassBuilder {
public:
    RenderPassBuilder() = default;
    explicit RenderPassBuilder(const std::string& label);

    RenderPassBuilder(const RenderPassBuilder&) = delete;
    RenderPassBuilder& operator=(const RenderPassBuilder&) = delete;
    RenderPassBuilder(RenderPassBuilder&&) noexcept = default;
    RenderPassBuilder& operator=(RenderPassBuilder&&) noexcept = default;

    RenderPassBuilder&& AddColorAttachment(WGPUTextureView view, LoadOp load, StoreOp store,
                                            ClearColor clear = {0.0, 0.0, 0.0, 1.0}) &&;
    RenderPassBuilder&& SetDepthStencilAttachment(WGPUTextureView view, LoadOp load, StoreOp store,
                                                   float32 clear_depth = 1.0f) &&;

    void Execute(WGPUCommandEncoder encoder,
                 std::function<void(WGPURenderPassEncoder)> fn) &&;

private:
    struct ColorAttachmentData {
        WGPUTextureView view;
        LoadOp load;
        StoreOp store;
        ClearColor clear;
    };
    std::vector<ColorAttachmentData> color_attachments_;

    struct DepthAttachmentData {
        WGPUTextureView view;
        LoadOp load;
        StoreOp store;
        float32 clear_depth;
    };
    bool has_depth_ = false;
    DepthAttachmentData depth_attachment_;
    std::string label_;
};

}  // namespace render
}  // namespace mps
