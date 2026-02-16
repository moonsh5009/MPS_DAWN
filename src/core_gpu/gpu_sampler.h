#pragma once

#include "core_gpu/gpu_types.h"
#include <string>

// Forward-declare WebGPU handle type
struct WGPUSamplerImpl;  typedef WGPUSamplerImpl* WGPUSampler;

namespace mps {
namespace gpu {

struct SamplerConfig {
    AddressMode address_mode_u = AddressMode::ClampToEdge;
    AddressMode address_mode_v = AddressMode::ClampToEdge;
    AddressMode address_mode_w = AddressMode::ClampToEdge;
    FilterMode mag_filter = FilterMode::Linear;
    FilterMode min_filter = FilterMode::Linear;
    FilterMode mipmap_filter = FilterMode::Linear;
    float32 lod_min_clamp = 0.0f;
    float32 lod_max_clamp = 32.0f;
    uint16 max_anisotropy = 1;
    std::string label;
};

class GPUSampler {
public:
    explicit GPUSampler(const SamplerConfig& config = {});  // throws GPUException
    ~GPUSampler();

    // Move-only
    GPUSampler(GPUSampler&& other) noexcept;
    GPUSampler& operator=(GPUSampler&& other) noexcept;
    GPUSampler(const GPUSampler&) = delete;
    GPUSampler& operator=(const GPUSampler&) = delete;

    WGPUSampler GetHandle() const;

private:
    void Release();
    WGPUSampler handle_ = nullptr;
};

}  // namespace gpu
}  // namespace mps
