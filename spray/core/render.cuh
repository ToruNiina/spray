#ifndef SPRAY_CORE_RENDER_CUH
#define SPRAY_CORE_RENDER_CUH
#include <spray/geom/point.hpp>
#include <spray/geom/sphere.hpp>
#include <spray/core/buffer_array.hpp>
#include <spray/core/material.hpp>
#include <thrust/device_ptr.h>
#include <vector_types.h>

namespace spray
{
namespace core
{

__global__
void render_kernel(const std::size_t width, const std::size_t height,
        const float rwidth, const float rheight,
        const spray::geom::point location,
        const spray::geom::point lower_left,
        const spray::geom::point horizontal,
        const spray::geom::point vertical,
        const std::size_t        N,
        thrust::device_ptr<const spray::core::material> material,
        thrust::device_ptr<const spray::geom::sphere>   spheres,
        thrust::device_ptr<uchar4> img,
        thrust::device_ptr<std::uint32_t> first_hit
        );


} // core
} // spray
#endif// SPRAY_CORE_RENDER_CUH
