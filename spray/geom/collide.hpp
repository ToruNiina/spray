#ifndef SPRAY_GEOM_COLLIDE_HPP
#define SPRAY_GEOM_COLLIDE_HPP
#include <spray/util/cuda_math.hpp>
#include <spray/geom/point.hpp>
#include <spray/geom/ray.hpp>
#include <spray/geom/sphere.hpp>
#include <vector_types.h>
#include <vector_functions.h>

namespace spray
{
namespace geom
{

struct collision
{
    float  t; // ray position parameter. if no hits, becomes inf.
    float3 n; // normal vector
};

SPRAY_HOST_DEVICE
inline point normal(const collision& c) noexcept
{
    return make_point(c.n.x, c.n.y, c.n.z);
}


SPRAY_DEVICE
SPRAY_INLINE collision collide(const ray&  r, const sphere& sph,
    const float t_min, const float t_max = spray::util::inf()) noexcept
{
    const auto oc = r.origin - center(sph);
    const auto b  = dot(oc, r.direction);
    const auto c  = len_sq(oc) - radius(sph) * radius(sph);
    const auto d  = b * b - c;
    if(d < 0.0)
    {
        return collision{spray::util::inf(), make_float3(0.0f, 0.0f, 0.0f)};
    }
    const auto sqrt_d = sqrtf(d);
    const auto t1 = -b - sqrt_d;
    if(t_min <= t1 && t1 <= t_max)
    {
        const auto n = (ray_at(r, t1) - center(sph)) / radius(sph);
        return collision{t1, make_float3(X(n), Y(n), Z(n))};
    }
    const auto t2 = -b + sqrt_d;
    if(t_min <= t2 && t2 <= t_max)
    {
        const auto n = (ray_at(r, t2) - center(sph)) / radius(sph);
        return collision{t2, make_float3(X(n), Y(n), Z(n))};
    }
    return collision{spray::util::inf(), make_float3(0.0f, 0.0f, 0.0f)};
}

} // geom
} // spray
#endif// SPRAY_GEOM_COLLIDE_HPP
