#include <spray/cuda/cuda_assert.hpp>
#include <spray/cuda/show_image.hpp>
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
namespace cuda
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
        thrust::device_ptr<uchar4> img)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const std::size_t offset = x + y * blockDim.x * gridDim.x;
    if(offset >= width * height)
    {
        return;
    }
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
        const spray::core::color color  = mat.albedo;
        pixel = make_pixel(color);
    }
    img[offset] = pixel;
    return;
}

void render_impl(const dim3 blocks, const dim3 threads, const cudaStream_t stream,
                 const cudaArray_const_t& buf, std::size_t w, std::size_t h,
                 const spray::geom::point loc,
                 const spray::geom::point lower_left,
                 const spray::geom::point horizontal,
                 const spray::geom::point vertical,
                 const spray::core::world_base& wld_base)
{
    const auto& wld = dynamic_cast<spray::core::world const&>(wld_base);
    if(!wld.is_loaded())
    {
        wld.load();
    }
    thrust::device_vector<uchar4> img(w * h);

    render_kernel<<<blocks, threads, 0, stream>>>(w, h, 1.0f / w, 1.0f / h,
            loc, lower_left, horizontal, vertical, wld.device_spheres().size(),
            thrust::device_pointer_cast(wld.device_materials().data()),
            thrust::device_pointer_cast(wld.device_spheres().data()),
            thrust::device_pointer_cast(img.data())
            );

    show_image(blocks, threads, stream, buf, w, h,
               thrust::device_pointer_cast(img.data()));

    return;
}

} // cuda
} // spray
