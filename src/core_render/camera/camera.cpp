#include "core_render/camera/camera.h"
#include "core_util/math.h"
#include <cmath>

using namespace mps::util;

namespace mps {
namespace render {

Camera::Camera(const CameraConfig& config)
    : target_(config.target)
    , up_(config.up)
    , fov_(config.fov)
    , aspect_ratio_(config.aspect_ratio)
    , near_plane_(config.near_plane)
    , far_plane_(config.far_plane)
    , min_distance_(config.min_distance)
    , max_distance_(config.max_distance) {

    vec3 dir = config.position - config.target;
    distance_ = glm::length(dir);
    if (distance_ > 0.0f) {
        dir /= distance_;
    }
    yaw_ = std::atan2(dir.x, dir.z);
    pitch_ = std::asin(Clamp(dir.y, -1.0f, 1.0f));

    position_dirty_ = true;
    dirty_ = true;
}

void Camera::UpdatePosition() {
    position_.x = target_.x + distance_ * std::cos(pitch_) * std::sin(yaw_);
    position_.y = target_.y + distance_ * std::sin(pitch_);
    position_.z = target_.z + distance_ * std::cos(pitch_) * std::cos(yaw_);
    position_dirty_ = false;
}

mat4 Camera::GetViewMatrix() const {
    return LookAt(GetPosition(), target_, up_);
}

mat4 Camera::GetProjectionMatrix() const {
    return Perspective(Radians(fov_), aspect_ratio_, near_plane_, far_plane_);
}

void Camera::Orbit(float32 delta_yaw, float32 delta_pitch) {
    yaw_ += delta_yaw;
    pitch_ += delta_pitch;

    // Clamp pitch to +/- 89 degrees
    const float32 max_pitch = Radians(89.0f);
    pitch_ = Clamp(pitch_, -max_pitch, max_pitch);

    position_dirty_ = true;
    dirty_ = true;
}

void Camera::Pan(float32 delta_x, float32 delta_y) {
    mat4 view = GetViewMatrix();
    vec3 right = vec3(view[0][0], view[1][0], view[2][0]);
    vec3 up = vec3(view[0][1], view[1][1], view[2][1]);

    target_ += right * delta_x + up * delta_y;

    position_dirty_ = true;
    dirty_ = true;
}

void Camera::Zoom(float32 delta) {
    distance_ *= (1.0f - delta * 0.1f);
    distance_ = Clamp(distance_, min_distance_, max_distance_);

    position_dirty_ = true;
    dirty_ = true;
}

void Camera::SetAspectRatio(float32 aspect) {
    aspect_ratio_ = aspect;
    dirty_ = true;
}

vec3 Camera::GetPosition() const {
    if (position_dirty_) {
        const_cast<Camera*>(this)->UpdatePosition();
    }
    return position_;
}

vec3 Camera::GetTarget() const {
    return target_;
}

float32 Camera::GetFov() const {
    return fov_;
}

float32 Camera::GetNearPlane() const {
    return near_plane_;
}

float32 Camera::GetFarPlane() const {
    return far_plane_;
}

float32 Camera::GetAspectRatio() const {
    return aspect_ratio_;
}

bool Camera::IsDirty() const {
    return dirty_;
}

void Camera::ClearDirty() {
    dirty_ = false;
}

}  // namespace render
}  // namespace mps
