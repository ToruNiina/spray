#ifndef SPRAY_GEOM_SPHERE_HPP
#define SPRAY_GEOM_SPHERE_HPP
#include <spray/geom/point.hpp>

namespace spray
{
namespace geom
{

struct sphere
{
    SPRAY_HOST_DEVICE
    sphere(const point& center, const float radius) noexcept
        : data(make_float4(X(center), Y(center), Z(center), radius))
    {}
    sphere() = default;
    sphere(const sphere&) = default;
    sphere(sphere&&)      = default;
    sphere& operator=(const sphere&) = default;
    sphere& operator=(sphere&&)      = default;
    ~sphere() = default;

    float4 data;
};

SPRAY_HOST_DEVICE
inline float radius(const sphere& sph) noexcept
{
    return sph.data.w;
}
SPRAY_HOST_DEVICE
inline point center(const sphere& sph) noexcept
{
    return make_point(sph.data.x, sph.data.y, sph.data.z);
}

} // geom
} // spray
#endif// SPRAY_GEOM_SPHERE_HPP
