#pragma once

#include "core_util/types.h"
#include <string>

// Forward-declare WebGPU handle type
struct WGPUShaderModuleImpl;  typedef WGPUShaderModuleImpl* WGPUShaderModule;

namespace mps {
namespace gpu {

struct ShaderConfig {
    std::string code;    // WGSL source
    std::string label;
};

class GPUShader {
public:
    explicit GPUShader(const ShaderConfig& config);  // throws GPUException
    ~GPUShader();

    // Move-only
    GPUShader(GPUShader&& other) noexcept;
    GPUShader& operator=(GPUShader&& other) noexcept;
    GPUShader(const GPUShader&) = delete;
    GPUShader& operator=(const GPUShader&) = delete;

    WGPUShaderModule GetHandle() const;

private:
    void Release();
    WGPUShaderModule handle_ = nullptr;
};

}  // namespace gpu
}  // namespace mps
