#ifndef SPRAY_CORE_PATH_TRACING_CUH
#define SPRAY_CORE_PATH_TRACING_CUH
#include <spray/util/cuda_util.hpp>
#include <spray/geom/sphere.hpp>
#include <spray/geom/ray.hpp>
#include <spray/geom/collide.hpp>
#include <spray/core/color.hpp>
#include <spray/core/material.hpp>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <cstdint>

namespace spray
{
namespace core
{

__device__ __inline__
thrust::pair<uchar4, std::uint32_t>
path_trace(spray::geom::ray    ray,
           const uchar4        background,
           const std::size_t   N,
           thrust::device_ptr<const spray::core::material> material,
           thrust::device_ptr<const spray::geom::sphere>   spheres)
{
    std::uint32_t index = 0xFFFFFFFF;
    spray::geom::collision col;
    col.t = spray::util::inf();
    for(std::size_t i=0; i<N; ++i)
    {
        const spray::geom::collision c = collide(ray, spheres[i], 0.0f);
        if(!isinf(c.t) && c.t < col.t)
        {
            index = i;
            col   = c;
        }
    }

    uchar4 pixel = background;
    if(index != 0xFFFFFFFF)
    {
        const spray::core::material mat = material[index];
        const spray::core::color    clr = mat.albedo * fabsf(spray::geom::dot(
                spray::geom::direction(ray), spray::geom::normal(col)));
        pixel = make_pixel(clr);
        pixel.w = 0xFF;
    }

    return thrust::make_pair(pixel, index);
}

} // core
} // spray
#endif// SPRAY_CORE_PATH_TRACING_CUH
