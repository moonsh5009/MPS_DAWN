#pragma once

#include "core_util/types.h"
#include "core_util/math.h"

namespace mps {
namespace render {

struct CameraConfig {
    util::vec3 position = {0.0f, 2.0f, 5.0f};
    util::vec3 target = {0.0f, 0.0f, 0.0f};
    util::vec3 up = {0.0f, 1.0f, 0.0f};
    float32 fov = 45.0f;  // degrees
    float32 aspect_ratio = 16.0f / 9.0f;
    float32 near_plane = 0.1f;
    float32 far_plane = 100.0f;
    float32 min_distance = 0.5f;
    float32 max_distance = 50.0f;
};

class Camera {
public:
    explicit Camera(const CameraConfig& config = {});

    util::mat4 GetViewMatrix() const;
    util::mat4 GetProjectionMatrix() const;

    void Orbit(float32 delta_yaw, float32 delta_pitch);
    void Pan(float32 delta_x, float32 delta_y);
    void Zoom(float32 delta);
    void SetAspectRatio(float32 aspect);

    util::vec3 GetPosition() const;
    util::vec3 GetTarget() const;
    float32 GetFov() const;
    float32 GetNearPlane() const;
    float32 GetFarPlane() const;
    float32 GetAspectRatio() const;

    bool IsDirty() const;
    void ClearDirty();

private:
    void UpdatePosition();

    util::vec3 target_;
    util::vec3 up_;
    float32 yaw_;      // horizontal angle (radians)
    float32 pitch_;    // vertical angle (radians)
    float32 distance_; // distance from target

    float32 fov_;
    float32 aspect_ratio_;
    float32 near_plane_;
    float32 far_plane_;
    float32 min_distance_;
    float32 max_distance_;

    mutable util::vec3 position_;
    mutable bool position_dirty_ = true;
    bool dirty_ = true;
};

}  // namespace render
}  // namespace mps
