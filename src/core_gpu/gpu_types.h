#pragma once

#include "core_util/types.h"
#include <stdexcept>

namespace mps {
namespace gpu {

// GPU-specific exception
class GPUException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// --- Flag enums (uint32, bitwise combinable, values match WGPU) ---

enum class BufferUsage : uint32 {
    None         = 0x00,
    MapRead      = 0x01,
    MapWrite     = 0x02,
    CopySrc      = 0x04,
    CopyDst      = 0x08,
    Index        = 0x10,
    Vertex       = 0x20,
    Uniform      = 0x40,
    Storage      = 0x80,
    Indirect     = 0x100,
    QueryResolve = 0x200,
};

inline constexpr BufferUsage operator|(BufferUsage a, BufferUsage b) {
    return static_cast<BufferUsage>(static_cast<uint32>(a) | static_cast<uint32>(b));
}

inline constexpr BufferUsage operator&(BufferUsage a, BufferUsage b) {
    return static_cast<BufferUsage>(static_cast<uint32>(a) & static_cast<uint32>(b));
}

enum class TextureUsage : uint32 {
    None             = 0x00,
    CopySrc          = 0x01,
    CopyDst          = 0x02,
    TextureBinding   = 0x04,
    StorageBinding   = 0x08,
    RenderAttachment = 0x10,
};

inline constexpr TextureUsage operator|(TextureUsage a, TextureUsage b) {
    return static_cast<TextureUsage>(static_cast<uint32>(a) | static_cast<uint32>(b));
}

inline constexpr TextureUsage operator&(TextureUsage a, TextureUsage b) {
    return static_cast<TextureUsage>(static_cast<uint32>(a) & static_cast<uint32>(b));
}

// --- Value enums (values match WGPU) ---

enum class TextureFormat : uint32 {
    Undefined       = 0x00,
    R8Unorm         = 0x01,
    R8Snorm         = 0x02,
    R8Uint          = 0x03,
    R8Sint          = 0x04,
    R16Float        = 0x09,
    RG8Unorm        = 0x0A,
    RG8Snorm        = 0x0B,
    R32Float        = 0x0E,
    R32Uint         = 0x0F,
    R32Sint         = 0x10,
    RG16Float       = 0x15,
    RGBA8Unorm      = 0x16,
    RGBA8UnormSrgb  = 0x17,
    RGBA8Snorm      = 0x18,
    RGBA8Uint       = 0x19,
    RGBA8Sint       = 0x1A,
    BGRA8Unorm      = 0x1B,
    BGRA8UnormSrgb  = 0x1C,
    RGB10A2Unorm    = 0x1E,
    RG32Float       = 0x21,
    RGBA16Float     = 0x28,
    RGBA32Float     = 0x29,
    Depth16Unorm    = 0x2D,
    Depth24Plus     = 0x2E,
    Depth24PlusStencil8 = 0x2F,
    Depth32Float    = 0x30,
    Depth32FloatStencil8 = 0x31,
};

enum class TextureDimension : uint32 {
    D1 = 0x01,
    D2 = 0x02,
    D3 = 0x03,
};

enum class AddressMode : uint32 {
    ClampToEdge  = 0x01,
    Repeat       = 0x02,
    MirrorRepeat = 0x03,
};

enum class FilterMode : uint32 {
    Nearest = 0x01,
    Linear  = 0x02,
};

enum class CompareFunction : uint32 {
    Undefined    = 0x00,
    Never        = 0x01,
    Less         = 0x02,
    Equal        = 0x03,
    LessEqual    = 0x04,
    Greater      = 0x05,
    NotEqual     = 0x06,
    GreaterEqual = 0x07,
    Always       = 0x08,
};

// --- Pipeline/render enums (values match WGPU) ---

enum class ShaderStage : uint32 {
    None     = 0x00,
    Vertex   = 0x01,
    Fragment = 0x02,
    Compute  = 0x04,
};

inline constexpr ShaderStage operator|(ShaderStage a, ShaderStage b) {
    return static_cast<ShaderStage>(static_cast<uint32>(a) | static_cast<uint32>(b));
}

enum class BindingType : uint32 {
    Uniform,
    Storage,
    ReadOnlyStorage,
    Sampler,
    FilteringSampler,
    Texture2D,
    StorageTexture2D,
};

enum class VertexFormat : uint32 {
    Float32    = 0x0E,
    Float32x2  = 0x0F,
    Float32x3  = 0x10,
    Float32x4  = 0x11,
    Sint32     = 0x16,
    Sint32x2   = 0x17,
    Sint32x3   = 0x18,
    Sint32x4   = 0x19,
    Uint32     = 0x1A,
    Uint32x2   = 0x1B,
    Uint32x3   = 0x1C,
    Uint32x4   = 0x1D,
    Unorm8x2   = 0x01,
    Unorm8x4   = 0x03,
};

enum class VertexStepMode : uint32 {
    Vertex   = 0x01,
    Instance = 0x02,
};

enum class PrimitiveTopology : uint32 {
    PointList     = 0x01,
    LineList      = 0x02,
    LineStrip     = 0x03,
    TriangleList  = 0x04,
    TriangleStrip = 0x05,
};

}  // namespace gpu
}  // namespace mps
