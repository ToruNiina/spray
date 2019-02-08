#ifndef SPRAY_GEOM_RAY_HPP
#define SPRAY_GEOM_RAY_HPP
#include <spray/geom/point.cuh>

namespace spray
{
namespace geom
{

struct ray
{
    point org;
    point dir;
};

__host__ __device__ inline point const& origin(const ray& r) noexcept {return r.org;}
__host__ __device__ inline point&       origin(ray& r)       noexcept {return r.org;}
__host__ __device__ inline point const& direction(const ray& r) noexcept {return r.dir;}
__host__ __device__ inline point&       direction(ray& r)       noexcept {return r.dir;}

__host__ __device__
inline ray make_ray(const point& origin, const point& direction) noexcept
{
    return ray{origin, direction * (1.0f / length(direction))};
}

__host__ __device__
inline point ray_at(const ray& r, const float t) noexcept
{
    return r.org + r.dir * t;
}


} // geom
} // spray
#endif// SPRAY_GEOM_RAY_HPP
