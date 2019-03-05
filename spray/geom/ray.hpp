#ifndef SPRAY_GEOM_RAY_HPP
#define SPRAY_GEOM_RAY_HPP
#include <spray/geom/point.hpp>

namespace spray
{
namespace geom
{

struct ray
{
    point origin;
    point direction;
};

SPRAY_HOST_DEVICE
inline ray make_ray(const point& ori, const point& dir) noexcept
{
    return ray{ori, unit(dir)};
}

SPRAY_HOST_DEVICE
inline point ray_at(const ray& r, const float t) noexcept
{
    return r.origin + r.direction * t;
}

} // geom
} // spray
#endif// SPRAY_GEOM_RAY_HPP
