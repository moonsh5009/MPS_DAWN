#include "core_gpu/gpu_shader.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/gpu_types.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <cassert>
#include <utility>

using namespace mps::util;

namespace mps {
namespace gpu {

// -- Construction -------------------------------------------------------------

GPUShader::GPUShader(const ShaderConfig& config) {
    auto& core = GPUCore::GetInstance();
    assert(core.IsInitialized());

    WGPUShaderSourceWGSL wgsl_source = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl_source.code = {config.code.data(), config.code.size()};

    WGPUShaderModuleDescriptor desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    desc.nextInChain = &wgsl_source.chain;
    desc.label = {config.label.data(), config.label.size()};

    handle_ = wgpuDeviceCreateShaderModule(core.GetDevice(), &desc);
    if (!handle_) {
        throw GPUException("Failed to create shader module: " + config.label);
    }

    LogInfo("GPUShader created: ", config.label);
}

GPUShader::~GPUShader() {
    Release();
}

// -- Move semantics -----------------------------------------------------------

GPUShader::GPUShader(GPUShader&& other) noexcept
    : handle_(other.handle_) {
    other.handle_ = nullptr;
}

GPUShader& GPUShader::operator=(GPUShader&& other) noexcept {
    if (this != &other) {
        Release();
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

// -- Accessors ----------------------------------------------------------------

WGPUShaderModule GPUShader::GetHandle() const { return handle_; }

// -- Internal -----------------------------------------------------------------

void GPUShader::Release() {
    if (handle_) {
        wgpuShaderModuleRelease(handle_);
        handle_ = nullptr;
    }
}

}  // namespace gpu
}  // namespace mps
