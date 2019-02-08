#ifndef SPRAY_GEOM_SPHERE_HPP
#define SPRAY_GEOM_SPHERE_HPP
#include <spray/geom/point.cuh>

namespace spray
{
namespace geom
{

struct sphere
{
    float4 data;
};

__host__ __device__
inline float radius(const sphere& sph) noexcept
{
    return sph.data.w;
}
__host__ __device__
inline point center(const sphere& sph) noexcept
{
    return point{float4{sph.data.x, sph.data.y, sph.data.z, 0.0f}};
}

} // geom
} // spray
#endif// SPRAY_GEOM_SPHERE_HPP
