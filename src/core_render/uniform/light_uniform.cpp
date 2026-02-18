#include "core_render/uniform/light_uniform.h"
#include "core_gpu/gpu_buffer.h"
#include "core_gpu/gpu_types.h"
#include <span>

using namespace mps::util;
using namespace mps::gpu;

namespace mps {
namespace render {

LightUniform::LightUniform() = default;
LightUniform::~LightUniform() = default;
LightUniform::LightUniform(LightUniform&&) noexcept = default;
LightUniform& LightUniform::operator=(LightUniform&&) noexcept = default;

void LightUniform::Initialize() {
    BufferConfig config;
    config.usage = BufferUsage::Uniform | BufferUsage::CopyDst;
    config.size = sizeof(LightUBOData);
    config.label = "light_uniform";

    buffer_ = std::make_unique<GPUBuffer<LightUBOData>>(config);

    // Set default values
    data_.direction = vec4(Normalize(vec3(-0.5f, -1.0f, -0.3f)), 0.0f);
    data_.ambient = vec4(0.15f, 0.15f, 0.15f, 1.0f);
    data_.diffuse = vec4(1.0f, 1.0f, 1.0f, 1.0f);
    data_.specular = vec4(1.0f, 1.0f, 1.0f, 32.0f);
    dirty_ = true;
}

bool LightUniform::Update() {
    if (!dirty_) {
        return false;
    }

    buffer_->WriteData(std::span<const LightUBOData>(&data_, 1));
    dirty_ = false;
    return true;
}

WGPUBuffer LightUniform::GetBuffer() const {
    return buffer_->GetHandle();
}

void LightUniform::SetDirection(const vec3& dir) {
    data_.direction = vec4(Normalize(dir), 0.0f);
    dirty_ = true;
}

void LightUniform::SetAmbient(const vec3& color, float32 intensity) {
    data_.ambient = vec4(color, intensity);
    dirty_ = true;
}

void LightUniform::SetDiffuse(const vec3& color, float32 intensity) {
    data_.diffuse = vec4(color, intensity);
    dirty_ = true;
}

void LightUniform::SetSpecular(const vec3& color, float32 shininess) {
    data_.specular = vec4(color, shininess);
    dirty_ = true;
}

}  // namespace render
}  // namespace mps
