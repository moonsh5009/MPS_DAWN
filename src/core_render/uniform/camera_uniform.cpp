#include "core_render/uniform/camera_uniform.h"
#include "core_render/camera/camera.h"
#include "core_gpu/gpu_buffer.h"
#include "core_gpu/gpu_types.h"
#include <glm/glm.hpp>
#include <span>

using namespace mps::util;
using namespace mps::gpu;

namespace mps {
namespace render {

CameraUniform::CameraUniform() = default;
CameraUniform::~CameraUniform() = default;
CameraUniform::CameraUniform(CameraUniform&&) noexcept = default;
CameraUniform& CameraUniform::operator=(CameraUniform&&) noexcept = default;

void CameraUniform::Initialize() {
    BufferConfig config;
    config.usage = BufferUsage::Uniform | BufferUsage::CopyDst;
    config.size = sizeof(CameraUBOData);
    config.label = "camera_uniform";

    buffer_ = std::make_unique<GPUBuffer<CameraUBOData>>(config);
}

void CameraUniform::Update(Camera& camera, uint32 width, uint32 height) {
    if (!camera.IsDirty()) {
        return;
    }

    CameraUBOData data;
    data.view_mat = camera.GetViewMatrix();
    data.view_inv_mat = glm::inverse(data.view_mat);
    data.proj_mat = camera.GetProjectionMatrix();
    data.proj_inv_mat = glm::inverse(data.proj_mat);
    data.position = vec4(camera.GetPosition(), 0.0f);
    data.viewport = vec4(0.0f, 0.0f, static_cast<float32>(width), static_cast<float32>(height));
    data.frustum = vec2(camera.GetNearPlane(), camera.GetFarPlane());
    data.padding = vec2(0.0f);

    buffer_->WriteData(std::span<const CameraUBOData>(&data, 1));
    camera.ClearDirty();
}

WGPUBuffer CameraUniform::GetBuffer() const {
    return buffer_->GetHandle();
}

}  // namespace render
}  // namespace mps
