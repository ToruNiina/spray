#include <spray/core/cuda_assert.hpp>
#include <spray/core/show_image.cuh>
#include <spray/core/color.hpp>
#include <spray/core/material.hpp>
#include <spray/core/world.cuh>
#include <spray/geom/sphere.hpp>
#include <spray/geom/ray.hpp>
#include <spray/geom/collide.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/pair.h>

#include <vector_types.h>
#include <vector_functions.h>

namespace spray
{
namespace core
{

__device__
float fclampf(float x, float minimum, float maximum)
{
    return fminf(fmaxf(x, minimum), maximum);
}

__device__
uchar4 make_pixel(spray::core::color col)
{
    uchar4 pixel;
    pixel.x = std::uint8_t(fclampf(sqrtf(spray::core::R(col)) * 256, 0, 255));
    pixel.y = std::uint8_t(fclampf(sqrtf(spray::core::G(col)) * 256, 0, 255));
    pixel.z = std::uint8_t(fclampf(sqrtf(spray::core::B(col)) * 256, 0, 255));
    pixel.w = 0xFF;
    return pixel;
}

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
        thrust::device_ptr<std::uint32_t> first_hit_obj)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x >= width)  {return;}
    if(y >= height) {return;}
    const std::size_t offset = x + y * width;

    const spray::geom::point dst = lower_left +
                                   ((x+0.5f) *  rwidth) * horizontal +
                                   ((y+0.5f) * rheight) * vertical;
    const spray::geom::ray ray = spray::geom::make_ray(location, dst - location);

    std::uint32_t index = 0xFFFFFFFF;
    spray::geom::collision col;
    col.t = spray::geom::inf();
    for(std::size_t i=0; i<N; ++i)
    {
        const spray::geom::collision c = collide(ray, spheres[i], 0.0f);
        if(!isinf(c.t) && c.t < col.t)
        {
            index = i;
            col   = c;
        }
    }
    uchar4 pixel;
    if(index == 0xFFFFFFFF)
    {
        pixel.x = 0x00;
        pixel.y = 0x00;
        pixel.z = 0x00;
        pixel.w = 0x00;
    }
    else
    {
        const spray::core::material mat = material[index];
        const spray::core::color  color = mat.albedo * fabsf(spray::geom::dot(
                spray::geom::direction(ray), spray::geom::normal(col)));

        pixel = make_pixel(color);
    }
    img[offset] = pixel;
    first_hit_obj[offset] = index;
    return;
}

} // core
} // spray
