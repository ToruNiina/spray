#ifndef SPRAY_GEOM_POINT_HPP
#define SPRAY_GEOM_POINT_HPP
#include <spray/util/cuda_macro.hpp>
#include <vector_types.h>
#include <vector_functions.h>
#include <math.h>

namespace spray
{
namespace geom
{

struct point
{
    float4 data;
};

SPRAY_HOST_DEVICE
inline point make_point(float x, float y, float z, float w) noexcept
{
    return point{float4{x, y, z, w}};
}
SPRAY_HOST_DEVICE
inline point make_point(float x, float y, float z) noexcept
{
    return point{float4{x, y, z, 0.0f}};
}

SPRAY_HOST_DEVICE inline float X(const point& p) noexcept {return p.data.x;}
SPRAY_HOST_DEVICE inline float Y(const point& p) noexcept {return p.data.y;}
SPRAY_HOST_DEVICE inline float Z(const point& p) noexcept {return p.data.z;}
SPRAY_HOST_DEVICE inline float W(const point& p) noexcept {return p.data.w;}

SPRAY_HOST_DEVICE inline float& X(point& p) noexcept {return p.data.x;}
SPRAY_HOST_DEVICE inline float& Y(point& p) noexcept {return p.data.y;}
SPRAY_HOST_DEVICE inline float& Z(point& p) noexcept {return p.data.z;}
SPRAY_HOST_DEVICE inline float& W(point& p) noexcept {return p.data.w;}


SPRAY_HOST_DEVICE
inline point operator-(const point& lhs) noexcept
{
    return make_point(-lhs.data.x, -lhs.data.y, -lhs.data.z, -lhs.data.w);
}
SPRAY_HOST_DEVICE
inline point operator+(const point& lhs, const point& rhs) noexcept
{
    return make_point(lhs.data.x + rhs.data.x, lhs.data.y + rhs.data.y,
                      lhs.data.z + rhs.data.z, lhs.data.w + rhs.data.w);
}
SPRAY_HOST_DEVICE
inline point operator-(const point& lhs, const point& rhs) noexcept
{
    return make_point(lhs.data.x - rhs.data.x, lhs.data.y - rhs.data.y,
                      lhs.data.z - rhs.data.z, lhs.data.w - rhs.data.w);
}
SPRAY_HOST_DEVICE
inline point operator+(const point& lhs, const float rhs) noexcept
{
    return make_point(lhs.data.x + rhs, lhs.data.y + rhs,
                      lhs.data.z + rhs, lhs.data.w + rhs);
}
SPRAY_HOST_DEVICE
inline point operator-(const point& lhs, const float rhs) noexcept
{
    return make_point(lhs.data.x - rhs, lhs.data.y - rhs,
                      lhs.data.z - rhs, lhs.data.w - rhs);
}
SPRAY_HOST_DEVICE
inline point operator*(const point& lhs, const float rhs) noexcept
{
    return make_point(lhs.data.x * rhs, lhs.data.y * rhs,
                      lhs.data.z * rhs, lhs.data.w * rhs);
}
SPRAY_HOST_DEVICE
inline point operator*(const float& lhs, const point rhs) noexcept
{
    return make_point(lhs * rhs.data.x, lhs * rhs.data.y,
                      lhs * rhs.data.z, lhs * rhs.data.w);
}
SPRAY_HOST_DEVICE
inline point operator/(const point& lhs, const float rhs) noexcept
{
    return make_point(lhs.data.x / rhs, lhs.data.y / rhs,
                      lhs.data.z / rhs, lhs.data.w / rhs);
}

SPRAY_HOST_DEVICE
inline float dot(const point& lhs, const point& rhs) noexcept
{
    return lhs.data.x * rhs.data.x + lhs.data.y * rhs.data.y +
           lhs.data.z * rhs.data.z + lhs.data.w * rhs.data.w;
}
SPRAY_HOST_DEVICE
inline point cross(const point& lhs, const point& rhs) noexcept
{
    return make_point(lhs.data.y * rhs.data.z - lhs.data.z * rhs.data.y,
                      lhs.data.z * rhs.data.x - lhs.data.x * rhs.data.z,
                      lhs.data.x * rhs.data.y - lhs.data.y * rhs.data.x);
}
SPRAY_HOST_DEVICE
inline float len_sq(const point& lhs) noexcept
{
    return lhs.data.x * lhs.data.x + lhs.data.y * lhs.data.y +
           lhs.data.z * lhs.data.z;
}
SPRAY_HOST_DEVICE
inline float len(const point& lhs) noexcept
{
    return sqrtf(len_sq(lhs));
}
SPRAY_HOST_DEVICE
inline point unit(const point& lhs) noexcept
{
    return lhs / sqrtf(len_sq(lhs));
}

SPRAY_HOST_DEVICE
inline point
rotate(const point& vec, const float angle, const point& axis) noexcept
{
    // this assumes that the axis is already regularized.
    const float half_angle = angle * 0.5;
    const float sin_theta  = sinf(half_angle);

    // P, Q, R, S are quaternion.
    const float Q0 = cosf(half_angle);
    const float Q1 = X(axis) * sin_theta;
    const float Q2 = Y(axis) * sin_theta;
    const float Q3 = Z(axis) * sin_theta;

    const float P0 = 0.0f;
    const float P1 = X(vec);
    const float P2 = Y(vec);
    const float P3 = Z(vec);

    const float R0 =  Q0;
    const float R1 = -Q1;
    const float R2 = -Q2;
    const float R3 = -Q3;

    const float QP0 = Q0 * P0 - Q1 * P1 - Q2 * P2 - Q3 * P3;
    const float QP1 = Q0 * P1 + Q1 * P0 + Q2 * P3 - Q3 * P2;
    const float QP2 = Q0 * P2 - Q1 * P3 + Q2 * P0 + Q3 * P1;
    const float QP3 = Q0 * P3 + Q1 * P2 - Q2 * P1 + Q3 * P0;

//  const float S0 = QP0 * R0 - QP1 * R1 - QP2 * R2 - QP3 * R3;
    const float S1 = QP0 * R1 + QP1 * R0 + QP2 * R3 - QP3 * R2;
    const float S2 = QP0 * R2 - QP1 * R3 + QP2 * R0 + QP3 * R1;
    const float S3 = QP0 * R3 + QP1 * R2 - QP2 * R1 + QP3 * R0;

    return make_point(S1, S2, S3);
}

} // geom
} // spray
#endif// SPRAY_GEOM_POINT_HPP
