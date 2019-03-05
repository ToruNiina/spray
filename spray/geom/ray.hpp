#ifndef SPRAY_GEOM_RAY_HPP
#define SPRAY_GEOM_RAY_HPP
#include <spray/geom/point.hpp>

namespace spray
{
namespace geom
{

struct ray
{
    SPRAY_HOST_DEVICE
    ray(const point& ori, const point& dir) noexcept
        : origin(ori), direction(unit(dir))
    {}
    SPRAY_HOST_DEVICE ray() = default;
    SPRAY_HOST_DEVICE ray(const ray&) = default;
    SPRAY_HOST_DEVICE ray(ray&&)      = default;
    SPRAY_HOST_DEVICE ray& operator=(const ray&) = default;
    SPRAY_HOST_DEVICE ray& operator=(ray&&)      = default;
    SPRAY_HOST_DEVICE ~ray() = default;

    point origin;
    point direction;
};

SPRAY_HOST_DEVICE
inline point ray_at(const ray& r, const float t) noexcept
{
    return r.origin + r.direction * t;
}

} // geom
} // spray
#endif// SPRAY_GEOM_RAY_HPP
