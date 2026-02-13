#pragma once

#include "core_util/types.h"

// GLM core
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

namespace mps {
namespace util {

// Vector types
using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;

using ivec2 = glm::ivec2;
using ivec3 = glm::ivec3;
using ivec4 = glm::ivec4;

using uvec2 = glm::uvec2;
using uvec3 = glm::uvec3;
using uvec4 = glm::uvec4;

// Matrix types
using mat2 = glm::mat2;
using mat3 = glm::mat3;
using mat4 = glm::mat4;

// Quaternion type
using quat = glm::quat;

// Common constants
constexpr float32 PI = 3.14159265358979323846f;
constexpr float32 TWO_PI = 2.0f * PI;
constexpr float32 HALF_PI = 0.5f * PI;
constexpr float32 DEG_TO_RAD = PI / 180.0f;
constexpr float32 RAD_TO_DEG = 180.0f / PI;

// Utility functions
inline float32 Radians(float32 degrees) {
    return glm::radians(degrees);
}

inline float32 Degrees(float32 radians) {
    return glm::degrees(radians);
}

template<typename T>
inline T Clamp(T value, T min, T max) {
    return glm::clamp(value, min, max);
}

template<typename T>
inline T Lerp(T a, T b, float32 t) {
    return glm::mix(a, b, t);
}

inline float32 Length(const vec3& v) {
    return glm::length(v);
}

inline vec3 Normalize(const vec3& v) {
    return glm::normalize(v);
}

inline float32 Dot(const vec3& a, const vec3& b) {
    return glm::dot(a, b);
}

inline vec3 Cross(const vec3& a, const vec3& b) {
    return glm::cross(a, b);
}

// Matrix operations
inline mat4 Translate(const mat4& m, const vec3& v) {
    return glm::translate(m, v);
}

inline mat4 Rotate(const mat4& m, float32 angle, const vec3& axis) {
    return glm::rotate(m, angle, axis);
}

inline mat4 Scale(const mat4& m, const vec3& v) {
    return glm::scale(m, v);
}

inline mat4 LookAt(const vec3& eye, const vec3& center, const vec3& up) {
    return glm::lookAt(eye, center, up);
}

inline mat4 Perspective(float32 fovy, float32 aspect, float32 near, float32 far) {
    return glm::perspective(fovy, aspect, near, far);
}

inline mat4 Ortho(float32 left, float32 right, float32 bottom, float32 top, float32 near, float32 far) {
    return glm::ortho(left, right, bottom, top, near, far);
}

}  // namespace util
}  // namespace mps
