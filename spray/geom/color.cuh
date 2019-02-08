#ifndef SPRAY_GEOM_COLOR_HPP
#define SPRAY_GEOM_COLOR_HPP
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

__device__ __host__
inline color make_color(float x, float y, float z, float w) noexcept
{
    return color{float4{x, y, z, w}};
}
__device__ __host__
inline color make_color(float x, float y, float z) noexcept
{
    return color{float4{x, y, z, 0.0f}};
}

__device__ __host__ inline float R(const color& c) noexcept {return c.rgba.x;}
__device__ __host__ inline float G(const color& c) noexcept {return c.rgba.y;}
__device__ __host__ inline float B(const color& c) noexcept {return c.rgba.z;}
__device__ __host__ inline float A(const color& c) noexcept {return c.rgba.w;}

__device__ __host__ inline float& R(color& c) noexcept {return c.rgba.x;}
__device__ __host__ inline float& G(color& c) noexcept {return c.rgba.y;}
__device__ __host__ inline float& B(color& c) noexcept {return c.rgba.z;}
__device__ __host__ inline float& A(color& c) noexcept {return c.rgba.w;}

__device__ __host__
inline color operator+(const color& lhs, const color& rhs) noexcept
{
    return make_color(lhs.rgba.x + rhs.rgba.x, lhs.rgba.y + rhs.rgba.y,
                      lhs.rgba.z + rhs.rgba.z, lhs.rgba.w + rhs.rgba.w);
}

__device__ __host__
inline color operator-(const color& lhs.rgba, const color& rhs.rgba) noexcept
{
    return make_color(lhs.rgba.x - rhs.rgba.x, lhs.rgba.y - rhs.rgba.y,
                      lhs.rgba.z - rhs.rgba.z, lhs.rgba.w - rhs.rgba.w);
}

__device__ __host__
inline color operator*(const color& lhs.rgba, const color& rhs.rgba) noexcept
{
    return make_color(lhs.rgba.x * rhs.rgba.x, lhs.rgba.y * rhs.rgba.y,
                      lhs.rgba.z * rhs.rgba.z, lhs.rgba.w * rhs.rgba.w);
}

__device__ __host__
inline color operator/(const color& lhs.rgba, const color& rhs.rgba) noexcept
{
    return make_color(lhs.rgba.x / rhs.rgba.x, lhs.rgba.y / rhs.rgba.y,
                      lhs.rgba.z / rhs.rgba.z, lhs.rgba.w / rhs.rgba.w);
}

} // geom
} // spray
#endif// SPRAY_GEOM_COLOR_HPP
