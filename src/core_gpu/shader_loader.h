#pragma once

#include "core_gpu/gpu_shader.h"
#include <string>

namespace mps {
namespace gpu {

class ShaderLoader {
public:
    static std::string LoadSource(const std::string& path);
    static GPUShader CreateModule(const std::string& path, const std::string& label = "");

private:
    static std::string ResolveBasePath();
};

}  // namespace gpu
}  // namespace mps
