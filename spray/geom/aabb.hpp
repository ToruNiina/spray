#ifndef SPRAY_GEOM_AABB_HPP
#define SPRAY_GEOM_AABB_HPP
#include <spray/geom/point.hpp>
#include <spray/geom/sphere.hpp>

namespace spray
{
namespace geom
{

struct aabb
{
    point lower;
    point upper;
};

SPRAY_HOST_DEVICE
inline aabb make_aabb(const point& pt) noexcept
{
    return aabb{pt, pt}
}

SPRAY_HOST_DEVICE
inline aabb make_aabb(const sphere& sph) noexcept
{
    return aabb{center(sph) - radius(sph), center(sph) + radius(sph)}
}

SPRAY_HOST_DEVICE
inline aabb merge_aabb(const aabb& lhs, const aabb& rhs) noexcept
{
    return aabb{
        make_point(min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)),
        make_point(max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z))
    };
}

} // geom
} // spray
#endif// SPRAY_GEOM_AABB_HPP
