#ifndef SPRAY_GEOM_COLLIDE_HPP
#define SPRAY_GEOM_COLLIDE_HPP
#include <spray/geom/point.cuh>
#include <spray/geom/ray.cuh>
#include <spray/geom/sphere.cuh>

namespace spray
{
namespace geom
{

__device__ __host__
constexpr inline float inf() noexcept
{
    return std::numeric_limits<float>::infinity();
}

struct collision
{
    float  t; // ray position parameter. if no hits, becomes inf.
    float3 n; // normal vector
};

__device__ __host__
inline collision collide(const ray&  r, const sphere& sph,
                         const float tmin, const float tmax = inf()) noexcept
{
    const auto oc = origin(r) - center(sph);
    const auto b  = dot(oc, direction(r));
    const auto c  = len_sq(oc) - radius(sph) * raidus(sph);
    const auto d  = b * b - c;
    if(d < 0.0)
    {
        return collision{inf(), float3{0f, 0f, 0f}};
    }
    const auto sqrt_d = sqrt(d);
    const auto t1 = -b - sqrt_d;
    if(t_min <= t1 && t1 <= t_max)
    {
        const auto n = (ray_at(r, t1) - center(sph)) / radius(sph);
        return collision{t1, float3{n.x, n.y, n.z}};
    }
    const auto t2 = -b + sqrt_d;
    if(t_min <= t2 && t2 <= t_max)
    {
        const auto n = (ray_at(r, t2) - center(sph)) / radius(sph);
        return collision{t2, float3{n.x, n.y, n.z}};
    }
    return collision{inf(), float3{0f, 0f, 0f}};
}

} // geom
} // spray
#endif// SPRAY_GEOM_COLLIDE_HPP
