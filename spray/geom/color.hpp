#ifndef SPRAY_GEOM_COLOR_HPP
#define SPRAY_GEOM_COLOR_HPP
#include <spray/cuda/cuda_macro.hpp>
#include <vector_types.h>

namespace spray
{
namespace geom
{

// XXX it does not consider alpha blending.
struct color
{
    float4 rgba;
};

SPRAY_HOST_DEVICE
inline color make_color(float x, float y, float z, float w) noexcept
{
    return color{float4{x, y, z, w}};
}
SPRAY_HOST_DEVICE
inline color make_color(float x, float y, float z) noexcept
{
    return color{float4{x, y, z, 0.0f}};
}

SPRAY_HOST_DEVICE inline float R(const color& c) noexcept {return c.rgba.x;}
SPRAY_HOST_DEVICE inline float G(const color& c) noexcept {return c.rgba.y;}
SPRAY_HOST_DEVICE inline float B(const color& c) noexcept {return c.rgba.z;}
SPRAY_HOST_DEVICE inline float A(const color& c) noexcept {return c.rgba.w;}

SPRAY_HOST_DEVICE inline float& R(color& c) noexcept {return c.rgba.x;}
SPRAY_HOST_DEVICE inline float& G(color& c) noexcept {return c.rgba.y;}
SPRAY_HOST_DEVICE inline float& B(color& c) noexcept {return c.rgba.z;}
SPRAY_HOST_DEVICE inline float& A(color& c) noexcept {return c.rgba.w;}

SPRAY_HOST_DEVICE
inline color operator+(const color& lhs, const color& rhs) noexcept
{
    return make_color(lhs.rgba.x + rhs.rgba.x, lhs.rgba.y + rhs.rgba.y,
                      lhs.rgba.z + rhs.rgba.z, lhs.rgba.w + rhs.rgba.w);
}

SPRAY_HOST_DEVICE
inline color operator-(const color& lhs, const color& rhs) noexcept
{
    return make_color(lhs.rgba.x - rhs.rgba.x, lhs.rgba.y - rhs.rgba.y,
                      lhs.rgba.z - rhs.rgba.z, lhs.rgba.w - rhs.rgba.w);
}

SPRAY_HOST_DEVICE
inline color operator*(const color& lhs, const color& rhs) noexcept
{
    return make_color(lhs.rgba.x * rhs.rgba.x, lhs.rgba.y * rhs.rgba.y,
                      lhs.rgba.z * rhs.rgba.z, lhs.rgba.w * rhs.rgba.w);
}

SPRAY_HOST_DEVICE
inline color operator/(const color& lhs, const color& rhs) noexcept
{
    return make_color(lhs.rgba.x / rhs.rgba.x, lhs.rgba.y / rhs.rgba.y,
                      lhs.rgba.z / rhs.rgba.z, lhs.rgba.w / rhs.rgba.w);
}

} // geom
} // spray
#endif// SPRAY_GEOM_COLOR_HPP
