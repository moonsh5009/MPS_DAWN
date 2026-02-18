#pragma once

#include "core_util/types.h"
#include "core_util/math.h"
#include <memory>

// Forward declare
struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace mps {
namespace gpu { template<typename T> class GPUBuffer; }
namespace render {

class Camera;

struct CameraUBOData {
    util::mat4 view_mat;
    util::mat4 view_inv_mat;
    util::mat4 proj_mat;
    util::mat4 proj_inv_mat;
    util::vec4 position;   // w = 0
    util::vec4 viewport;   // x, y, width, height
    util::vec2 frustum;    // near, far
    util::vec2 padding;
};
static_assert(sizeof(CameraUBOData) == 304, "CameraUBOData must be 304 bytes");

class CameraUniform {
public:
    CameraUniform();
    ~CameraUniform();

    CameraUniform(CameraUniform&&) noexcept;
    CameraUniform& operator=(CameraUniform&&) noexcept;
    CameraUniform(const CameraUniform&) = delete;
    CameraUniform& operator=(const CameraUniform&) = delete;

    void Initialize();
    void Update(Camera& camera, uint32 width, uint32 height);
    WGPUBuffer GetBuffer() const;

private:
    std::unique_ptr<gpu::GPUBuffer<CameraUBOData>> buffer_;
};

}  // namespace render
}  // namespace mps
