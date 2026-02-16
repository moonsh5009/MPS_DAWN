#include "core_gpu/gpu_sampler.h"
#include "core_gpu/gpu_core.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <cassert>
#include <utility>

using namespace mps::util;

namespace mps {
namespace gpu {

// -- Construction -------------------------------------------------------------

GPUSampler::GPUSampler(const SamplerConfig& config) {
    auto& core = GPUCore::GetInstance();
    assert(core.IsInitialized());

    WGPUSamplerDescriptor desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    desc.label = {config.label.data(), config.label.size()};
    desc.addressModeU = static_cast<WGPUAddressMode>(config.address_mode_u);
    desc.addressModeV = static_cast<WGPUAddressMode>(config.address_mode_v);
    desc.addressModeW = static_cast<WGPUAddressMode>(config.address_mode_w);
    desc.magFilter = static_cast<WGPUFilterMode>(config.mag_filter);
    desc.minFilter = static_cast<WGPUFilterMode>(config.min_filter);
    desc.mipmapFilter = static_cast<WGPUMipmapFilterMode>(config.mipmap_filter);
    desc.lodMinClamp = config.lod_min_clamp;
    desc.lodMaxClamp = config.lod_max_clamp;
    desc.maxAnisotropy = config.max_anisotropy;

    handle_ = wgpuDeviceCreateSampler(core.GetDevice(), &desc);
    if (!handle_) {
        throw GPUException("Failed to create GPU sampler: " + config.label);
    }

    LogInfo("GPUSampler created: ", config.label);
}

GPUSampler::~GPUSampler() {
    Release();
}

// -- Move semantics -----------------------------------------------------------

GPUSampler::GPUSampler(GPUSampler&& other) noexcept
    : handle_(other.handle_) {
    other.handle_ = nullptr;
}

GPUSampler& GPUSampler::operator=(GPUSampler&& other) noexcept {
    if (this != &other) {
        Release();
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

// -- Accessors ----------------------------------------------------------------

WGPUSampler GPUSampler::GetHandle() const { return handle_; }

// -- Internal -----------------------------------------------------------------

void GPUSampler::Release() {
    if (handle_) {
        wgpuSamplerRelease(handle_);
        handle_ = nullptr;
    }
}

}  // namespace gpu
}  // namespace mps
