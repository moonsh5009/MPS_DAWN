#include "core_render/pass/render_pass_builder.h"
#include <webgpu/webgpu.h>

namespace mps {
namespace render {

RenderPassBuilder::RenderPassBuilder(const std::string& label)
    : label_(label) {}

RenderPassBuilder&& RenderPassBuilder::AddColorAttachment(WGPUTextureView view, LoadOp load,
                                                           StoreOp store, ClearColor clear) && {
    color_attachments_.push_back({view, load, store, clear});
    return std::move(*this);
}

RenderPassBuilder&& RenderPassBuilder::SetDepthStencilAttachment(WGPUTextureView view, LoadOp load,
                                                                  StoreOp store, float32 clear_depth) && {
    has_depth_ = true;
    depth_attachment_ = {view, load, store, clear_depth};
    return std::move(*this);
}

void RenderPassBuilder::Execute(WGPUCommandEncoder encoder,
                                 std::function<void(WGPURenderPassEncoder)> fn) && {
    // Build color attachments
    std::vector<WGPURenderPassColorAttachment> colors;
    for (const auto& ca : color_attachments_) {
        WGPURenderPassColorAttachment att = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        att.view = ca.view;
        att.loadOp = static_cast<WGPULoadOp>(ca.load);
        att.storeOp = static_cast<WGPUStoreOp>(ca.store);
        att.clearValue = {ca.clear.r, ca.clear.g, ca.clear.b, ca.clear.a};
        colors.push_back(att);
    }

    // Build depth attachment
    WGPURenderPassDepthStencilAttachment depth = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
    if (has_depth_) {
        depth.view = depth_attachment_.view;
        depth.depthLoadOp = static_cast<WGPULoadOp>(depth_attachment_.load);
        depth.depthStoreOp = static_cast<WGPUStoreOp>(depth_attachment_.store);
        depth.depthClearValue = depth_attachment_.clear_depth;
    }

    // Build descriptor
    WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    desc.label = {label_.data(), label_.size()};
    desc.colorAttachmentCount = colors.size();
    desc.colorAttachments = colors.data();
    if (has_depth_) {
        desc.depthStencilAttachment = &depth;
    }

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &desc);
    fn(pass);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
}

}  // namespace render
}  // namespace mps
