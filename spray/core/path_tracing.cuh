#ifndef SPRAY_CORE_PATH_TRACING_CUH
#define SPRAY_CORE_PATH_TRACING_CUH
#include <spray/util/cuda_util.hpp>
#include <spray/geom/sphere.hpp>
#include <spray/geom/ray.hpp>
#include <spray/geom/collide.hpp>
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

// {pixel, first-hit-object, next-seed}
__device__ __inline__
thrust::tuple<uchar4, std::uint32_t, std::uint32_t>
path_trace(const spray::geom::ray   ray,
           const spray::core::color background,
           const std::uint32_t      seed,
           const std::uint32_t      depth,
           const std::size_t        N,
           const thrust::device_ptr<const spray::core::material> material,
           const thrust::device_ptr<const spray::geom::sphere>   spheres)
{
    thrust::default_random_engine rng(seed);
    uchar4 pixel = make_pixel(background);

    std::uint32_t index;
    spray::geom::collision col;
    thrust::tie(col, index) = hit(ray, 0.0f, N, spheres);

    if(index != 0xFFFFFFFF)
    {
        const spray::core::material mat = material[index];
        const spray::core::color    clr = mat.albedo * fabsf(spray::geom::dot(
                spray::geom::direction(ray), spray::geom::normal(col)));
        pixel = make_pixel(clr);
        pixel.w = 0xFF;
    }

    return thrust::make_tuple(pixel, index, rng());
}

} // core
} // spray
#endif// SPRAY_CORE_PATH_TRACING_CUH
