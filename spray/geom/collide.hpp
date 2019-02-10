#ifndef SPRAY_GEOM_COLLIDE_HPP
#define SPRAY_GEOM_COLLIDE_HPP
#include <spray/geom/point.hpp>
#include <spray/geom/ray.hpp>
#include <spray/geom/sphere.hpp>
#include <math_constants.h>
#include <vector_types.h>
#include <vector_functions.h>

namespace spray
{
namespace geom
{

#ifdef __CUDACC__
__device__
inline float inf() noexcept
{
    return CUDART_INF_F;
}
#else
inline float inf() noexcept
{
    return std::numeric_limits<float>::infinity();
}
#endif

struct collision
{
    float  t; // ray position parameter. if no hits, becomes inf.
    float3 n; // normal vector
};

SPRAY_DEVICE
inline collision collide(const ray&  r, const sphere& sph,
                         const float t_min, const float t_max = inf()) noexcept
{
    const auto oc = origin(r) - center(sph);
    const auto b  = dot(oc, direction(r));
    const auto c  = len_sq(oc) - radius(sph) * radius(sph);
    const auto d  = b * b - c;
    if(d < 0.0)
    {
        return collision{inf(), make_float3(0.0f, 0.0f, 0.0f)};
    }
    const auto sqrt_d = sqrt(d);
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
    return collision{inf(), make_float3(0.0f, 0.0f, 0.0f)};
}

} // geom
} // spray
#endif// SPRAY_GEOM_COLLIDE_HPP
