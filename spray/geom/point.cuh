#ifndef SPRAY_GEOM_POINT_HPP
#define SPRAY_GEOM_POINT_HPP
#include <vector_types.h>

namespace spray
{
namespace geom
{

struct point
{
    float4 data;
};

__device__ __host__
inline point make_point(float x, float y, float z, float w) noexcept
{
    return point{float4{x, y, z, w}};
}
__device__ __host__
inline point make_point(float x, float y, float z) noexcept
{
    return point{float4{x, y, z, 0.0f}};
}

__device__ __host__ inline float X(const point& p) noexcept {return p.data.x;}
__device__ __host__ inline float Y(const point& p) noexcept {return p.data.y;}
__device__ __host__ inline float Z(const point& p) noexcept {return p.data.z;}
__device__ __host__ inline float W(const point& p) noexcept {return p.data.w;}

__device__ __host__ inline float& X(point& p) noexcept {return p.data.x;}
__device__ __host__ inline float& Y(point& p) noexcept {return p.data.y;}
__device__ __host__ inline float& Z(point& p) noexcept {return p.data.z;}
__device__ __host__ inline float& W(point& p) noexcept {return p.data.w;}


__device__ __host__
inline point operator-(const point& lhs) noexcept
{
    return make_point(-lhs.data.x, -lhs.data.y, -lhs.data.z, -lhs.data.w);
}
__device__ __host__
inline point operator+(const point& lhs, const point& rhs) noexcept
{
    return make_point(lhs.data.x + rhs.data.x, lhs.data.y + rhs.data.y,
                      lhs.data.z + rhs.data.z, lhs.data.w + rhs.data.w);
}
__device__ __host__
inline point operator-(const point& lhs, const point& rhs) noexcept
{
    return make_point(lhs.data.x - rhs.data.x, lhs.data.y - rhs.data.y,
                      lhs.data.z - rhs.data.z, lhs.data.w - rhs.data.w);
}
__device__ __host__
inline point operator+(const point& lhs, const float rhs) noexcept
{
    return make_point(lhs.data.x + rhs, lhs.data.y + rhs,
                      lhs.data.z + rhs, lhs.data.w + rhs);
}
__device__ __host__
inline point operator-(const point& lhs, const float rhs) noexcept
{
    return make_point(lhs.data.x - rhs, lhs.data.y - rhs,
                      lhs.data.z - rhs, lhs.data.w - rhs);
}
__device__ __host__
inline point operator*(const point& lhs, const float rhs) noexcept
{
    return make_point(lhs.data.x * rhs, lhs.data.y * rhs,
                      lhs.data.z * rhs, lhs.data.w * rhs);
}
__device__ __host__
inline point operator*(const float& lhs, const point rhs) noexcept
{
    return make_point(lhs * rhs.data.x, lhs * rhs.data.y,
                      lhs * rhs.data.z, lhs * rhs.data.w);
}
__device__ __host__
inline point operator/(const point& lhs, const float rhs) noexcept
{
    return make_point(lhs.data.x / rhs, lhs.data.y / rhs,
                      lhs.data.z / rhs, lhs.data.w / rhs);
}

__device__ __host__
inline float dot(const point& lhs, const point& rhs) noexcept
{
    return lhs.data.x * rhs.data.x + lhs.data.y * rhs.data.y +
           lhs.data.z * rhs.data.z + lhs.data.w * rhs.data.w;
}
__device__ __host__
inline point cross(const point& lhs, const point& rhs) noexcept
{
    return make_point(lhs.data.y * rhs.data.z - lhs.data.z * rhs.data.y,
                      lhs.data.z * rhs.data.x - lhs.data.x * rhs.data.z,
                      lhs.data.x * rhs.data.y - lhs.data.y * rhs.data.x);
}
__device__ __host__
inline float len_sq(const point& lhs) noexcept
{
    return lhs.data.x * lhs.data.x + lhs.data.y * lhs.data.y +
           lhs.data.z * lhs.data.z;
}
__device__ __host__
inline float len(const point& lhs) noexcept
{
    return sqrt(len_sq(lhs));
}
__device__ __host__
inline point unit(const point& lhs) noexcept
{
    return lhs / len(lhs);
}

} // geom
} // spray
#endif// SPRAY_GEOM_POINT_HPP
