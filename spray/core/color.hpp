#ifndef SPRAY_CORE_COLOR_HPP
#define SPRAY_CORE_COLOR_HPP
#include <spray/util/cuda_math.hpp>
#include <vector_types.h>
#include <cstdint>

namespace spray
{
namespace core
{

// XXX it does not consider alpha blending.
struct color
{
    float4 rgba;
};

SPRAY_HOST_DEVICE
inline color make_color(float x, float y, float z, float w) noexcept
{
    return color{make_float4(x, y, z, w)};
}
SPRAY_HOST_DEVICE
inline color make_color(float x, float y, float z) noexcept
{
    return color{make_float4(x, y, z, 1.0f)};
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

SPRAY_HOST_DEVICE
inline color operator*(const float lhs, const color& rhs) noexcept
{
    return make_color(lhs * rhs.rgba.x, lhs * rhs.rgba.y,
                      lhs * rhs.rgba.z, lhs * rhs.rgba.w);
}

SPRAY_HOST_DEVICE
inline color operator*(const color& lhs, const float rhs) noexcept
{
    return make_color(lhs.rgba.x * rhs, lhs.rgba.y * rhs,
                      lhs.rgba.z * rhs, lhs.rgba.w * rhs);
}

SPRAY_HOST_DEVICE
inline color operator/(const color& lhs, const float rhs) noexcept
{
    return make_color(lhs.rgba.x / rhs, lhs.rgba.y / rhs,
                      lhs.rgba.z / rhs, lhs.rgba.w / rhs);
}

SPRAY_HOST_DEVICE
SPRAY_INLINE color gamma_correction(color col, float gamma) noexcept
{
    return make_color(powf(spray::core::R(col), 1.0f / gamma),
                      powf(spray::core::G(col), 1.0f / gamma),
                      powf(spray::core::B(col), 1.0f / gamma),
                      spray::core::A(col));
}
    
SPRAY_HOST_DEVICE
SPRAY_INLINE uchar4 make_pixel(spray::core::color col)
{
    col = gamma_correction(col, 2.2);
    uchar4 pixel;
    pixel.x = std::uint8_t(spray::util::fclampf(spray::core::R(col) * 256, 0, 255));
    pixel.y = std::uint8_t(spray::util::fclampf(spray::core::G(col) * 256, 0, 255));
    pixel.z = std::uint8_t(spray::util::fclampf(spray::core::B(col) * 256, 0, 255));
    pixel.w = std::uint8_t(spray::util::fclampf(spray::core::A(col) * 256, 0, 255));
    return pixel;
}


} // core
} // spray
#endif// SPRAY_CORE_COLOR_HPP
