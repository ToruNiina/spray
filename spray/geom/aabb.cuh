#ifndef SPRAY_GEOM_AABB_HPP
#define SPRAY_GEOM_AABB_HPP
#include <spray/geom/point.cuh>
#include <spray/geom/sphere.cuh>

namespace spray
{
namespace geom
{

struct aabb
{
    point lower;
    point upper;
};

__device__ __host__
inline aabb make_aabb(const point& pt) noexcept
{
    return aabb{pt, pt}
}

__device__ __host__
inline aabb make_aabb(const sphere& sph) noexcept
{
    return aabb{center(sph) - radius(sph), center(sph) + radius(sph)}
}

} // geom
} // spray
#endif// SPRAY_GEOM_AABB_HPP