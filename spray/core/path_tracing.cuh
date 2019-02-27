#ifndef SPRAY_CORE_PATH_TRACING_CUH
#define SPRAY_CORE_PATH_TRACING_CUH
#include <spray/util/cuda_math.hpp>
#include <spray/geom/sphere.hpp>
#include <spray/geom/ray.hpp>
#include <spray/geom/collide.hpp>
#include <spray/geom/lambertian.cuh>
#include <spray/core/color.hpp>
#include <spray/core/material.hpp>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <cstdint>

namespace spray
{
namespace core
{


__device__ __inline__
thrust::pair<spray::geom::collision, std::uint32_t>
hit(spray::geom::ray r, const float start, const std::size_t N,
    thrust::device_ptr<const spray::geom::sphere> spheres)
{
    std::uint32_t index = 0xFFFFFFFF;
    spray::geom::collision col;
    col.t = spray::util::inf();

    for(std::size_t i=0; i<N; ++i)
    {
        const spray::geom::collision c = collide(r, spheres[i], start);
        if(!isinf(c.t) && c.t < col.t)
        {
            index = i;
            col   = c;
        }
    }
    return thrust::make_pair(col, index);
}

/* i: intensity, a: albedo, e: emission
 *
 * i0 = e0 + a0 * i1
 *    = e0 + a0 * (e1 + a1 * i2)
 *    = e0 + a0 * (e1 + a1 * (e2 + a2 * i3))
 *    = e0 + a0 * (e1 + a1 * (e2 + a2 * (e3 + a3 * i4)))
 *    = ...
 * i0 = e0 + a0 * i1
 *    = e0 + a0 * e1 + a0 * a1 * i2
 *    = e0 + a0 * e1 + a0 * a1 * e2 + a0 * a1 * a2 * i3
 *    = e0 + a0 * e1 + a0 * a1 * e2 + a0 * a1 * a2 * e3 + a0 * a1 * a2 * a3 * i4
 *    = ...
 */

// {pixel, first-hit-object, next-seed}
template<typename RNG>
__device__ __inline__
thrust::pair<spray::core::color, std::uint32_t>
path_trace(const spray::geom::ray   ray,
           const spray::core::color background,
           const std::uint32_t      depth,
           const std::size_t        N,
           const thrust::device_ptr<const spray::core::material> material,
           const thrust::device_ptr<const spray::geom::sphere>   spheres,
           RNG& rng)
{
    std::uint32_t          index;
    spray::geom::collision col;
    thrust::tie(col, index) = spray::core::hit(ray, 0.0f, N, spheres);
    const std::uint32_t first_hit = index;

    spray::core::color intensity = spray::core::make_color(0.0f, 0.0f, 0.0f, 1.0f);
    spray::core::color albedo    = spray::core::make_color(1.0f, 1.0f, 1.0f, 1.0f);
    spray::geom::ray next_ray = ray;

    for(std::uint32_t i=0; i<depth; ++i)
    {
        if(index == 0xFFFFFFFF) {break;}

        const spray::core::material mat = material[index];
        intensity = intensity + albedo * mat.emission;
        albedo    =             albedo * mat.albedo;

        next_ray = spray::geom::scatter_lambertian(next_ray,
                spray::geom::ray_at(next_ray, col.t),
                spray::geom::make_point(col.n.x, col.n.y, col.n.z), rng);

        thrust::tie(col, index) = spray::core::hit(next_ray, 0.0f, N, spheres);
    }
    intensity = intensity + albedo * background;

    return thrust::make_pair(intensity, first_hit);
}


} // core
} // spray
#endif// SPRAY_CORE_PATH_TRACING_CUH
