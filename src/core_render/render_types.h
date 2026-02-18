#pragma once

#include "core_util/types.h"
#include "core_gpu/gpu_types.h"

// Forward-declare WebGPU handle types used by render module
struct WGPUSurfaceImpl;             typedef WGPUSurfaceImpl*             WGPUSurface;
struct WGPUTextureViewImpl;         typedef WGPUTextureViewImpl*         WGPUTextureView;
struct WGPUShaderModuleImpl;        typedef WGPUShaderModuleImpl*        WGPUShaderModule;
struct WGPURenderPipelineImpl;      typedef WGPURenderPipelineImpl*      WGPURenderPipeline;
struct WGPUPipelineLayoutImpl;      typedef WGPUPipelineLayoutImpl*      WGPUPipelineLayout;
struct WGPUBufferImpl;              typedef WGPUBufferImpl*              WGPUBuffer;
struct WGPUBindGroupImpl;           typedef WGPUBindGroupImpl*           WGPUBindGroup;
struct WGPUBindGroupLayoutImpl;     typedef WGPUBindGroupLayoutImpl*     WGPUBindGroupLayout;
struct WGPUSamplerImpl;             typedef WGPUSamplerImpl*             WGPUSampler;
struct WGPUCommandEncoderImpl;      typedef WGPUCommandEncoderImpl*      WGPUCommandEncoder;
struct WGPURenderPassEncoderImpl;   typedef WGPURenderPassEncoderImpl*   WGPURenderPassEncoder;
struct WGPURenderBundleEncoderImpl; typedef WGPURenderBundleEncoderImpl* WGPURenderBundleEncoder;
struct WGPUTextureImpl;             typedef WGPUTextureImpl*             WGPUTexture;

namespace mps {
namespace render {

// Cull mode (values match WGPUCullMode)
enum class CullMode : uint32 {
    None  = 0x01,
    Front = 0x02,
    Back  = 0x03,
};

// Front face winding (values match WGPUFrontFace)
enum class FrontFace : uint32 {
    CCW = 0x01,
    CW  = 0x02,
};

// Load operation (values match WGPULoadOp)
enum class LoadOp : uint32 {
    Undefined = 0x00,
    Load      = 0x01,
    Clear     = 0x02,
};

// Store operation (values match WGPUStoreOp)
enum class StoreOp : uint32 {
    Undefined = 0x00,
    Store     = 0x01,
    Discard   = 0x02,
};

// Blend factor (values match WGPUBlendFactor)
enum class BlendFactor : uint32 {
    Zero             = 0x01,
    One              = 0x02,
    Src              = 0x03,
    OneMinusSrc      = 0x04,
    SrcAlpha         = 0x05,
    OneMinusSrcAlpha = 0x06,
    Dst              = 0x07,
    OneMinusDst      = 0x08,
    DstAlpha         = 0x09,
    OneMinusDstAlpha = 0x0A,
};

// Blend operation (values match WGPUBlendOperation)
enum class BlendOp : uint32 {
    Add             = 0x01,
    Subtract        = 0x02,
    ReverseSubtract = 0x03,
    Min             = 0x04,
    Max             = 0x05,
};

struct BlendState {
    BlendFactor src_color = BlendFactor::SrcAlpha;
    BlendOp color_op = BlendOp::Add;
    BlendFactor dst_color = BlendFactor::OneMinusSrcAlpha;
    BlendFactor src_alpha = BlendFactor::One;
    BlendOp alpha_op = BlendOp::Add;
    BlendFactor dst_alpha = BlendFactor::OneMinusSrcAlpha;
};

struct ClearColor {
    float64 r = 0.0;
    float64 g = 0.0;
    float64 b = 0.0;
    float64 a = 1.0;
};

}  // namespace render
}  // namespace mps
