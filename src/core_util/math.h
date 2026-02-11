#pragma once

// GLM core
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
constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;
constexpr float HALF_PI = 0.5f * PI;
constexpr float DEG_TO_RAD = PI / 180.0f;
constexpr float RAD_TO_DEG = 180.0f / PI;

// Utility functions
inline float Radians(float degrees) {
    return glm::radians(degrees);
}

inline float Degrees(float radians) {
    return glm::degrees(radians);
}

template<typename T>
inline T Clamp(T value, T min, T max) {
    return glm::clamp(value, min, max);
}

template<typename T>
inline T Lerp(T a, T b, float t) {
    return glm::mix(a, b, t);
}

inline float Length(const vec3& v) {
    return glm::length(v);
}

inline vec3 Normalize(const vec3& v) {
    return glm::normalize(v);
}

inline float Dot(const vec3& a, const vec3& b) {
    return glm::dot(a, b);
}

inline vec3 Cross(const vec3& a, const vec3& b) {
    return glm::cross(a, b);
}

// Matrix operations
inline mat4 Translate(const mat4& m, const vec3& v) {
    return glm::translate(m, v);
}

inline mat4 Rotate(const mat4& m, float angle, const vec3& axis) {
    return glm::rotate(m, angle, axis);
}

inline mat4 Scale(const mat4& m, const vec3& v) {
    return glm::scale(m, v);
}

inline mat4 LookAt(const vec3& eye, const vec3& center, const vec3& up) {
    return glm::lookAt(eye, center, up);
}

inline mat4 Perspective(float fovy, float aspect, float near, float far) {
    return glm::perspective(fovy, aspect, near, far);
}

inline mat4 Ortho(float left, float right, float bottom, float top, float near, float far) {
    return glm::ortho(left, right, bottom, top, near, far);
}

}  // namespace util
}  // namespace mps
