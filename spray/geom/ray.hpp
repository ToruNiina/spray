#ifndef SPRAY_GEOM_RAY_HPP
#define SPRAY_GEOM_RAY_HPP
#include <spray/geom/point.hpp>

namespace spray
{
namespace geom
{

struct ray
{
    point org;
    point dir;
};

SPRAY_HOST_DEVICE inline point const& origin(const ray& r) noexcept {return r.org;}
SPRAY_HOST_DEVICE inline point&       origin(ray& r)       noexcept {return r.org;}
SPRAY_HOST_DEVICE inline point const& direction(const ray& r) noexcept {return r.dir;}
SPRAY_HOST_DEVICE inline point&       direction(ray& r)       noexcept {return r.dir;}

SPRAY_HOST_DEVICE
inline ray make_ray(const point& origin, const point& direction) noexcept
{
    return ray{origin, direction * (1.0f / length(direction))};
}

SPRAY_HOST_DEVICE
inline point ray_at(const ray& r, const float t) noexcept
{
    return r.org + r.dir * t;
}


} // geom
} // spray
#endif// SPRAY_GEOM_RAY_HPP
