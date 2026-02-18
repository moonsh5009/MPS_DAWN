#pragma once

#include "core_util/types.h"
#include "core_util/math.h"
#include <memory>

struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace mps {
namespace gpu { template<typename T> class GPUBuffer; }
namespace render {

struct LightUBOData {
    util::vec4 direction;   // xyz = direction, w = 0
    util::vec4 ambient;     // rgb = color, a = intensity
    util::vec4 diffuse;     // rgb = color, a = intensity
    util::vec4 specular;    // rgb = color, a = shininess
};

class LightUniform {
public:
    LightUniform();
    ~LightUniform();

    LightUniform(LightUniform&&) noexcept;
    LightUniform& operator=(LightUniform&&) noexcept;
    LightUniform(const LightUniform&) = delete;
    LightUniform& operator=(const LightUniform&) = delete;

    void Initialize();
    bool Update();
    WGPUBuffer GetBuffer() const;

    void SetDirection(const util::vec3& dir);
    void SetAmbient(const util::vec3& color, float32 intensity = 1.0f);
    void SetDiffuse(const util::vec3& color, float32 intensity = 1.0f);
    void SetSpecular(const util::vec3& color, float32 shininess = 32.0f);

private:
    std::unique_ptr<gpu::GPUBuffer<LightUBOData>> buffer_;
    LightUBOData data_;
    bool dirty_ = true;
};

}  // namespace render
}  // namespace mps
