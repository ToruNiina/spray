#ifndef SPRAY_CUDA_RENDER_HPP
#define SPRAY_CUDA_RENDER_HPP
#include <spray/core/world_base.hpp>
#include <spray/core/camera.hpp>
#include <spray/cuda/buffer_array.hpp>

namespace spray
{
namespace cuda
{

void render_impl(const dim3 blocks, const dim3 threads, const cudaStream_t stream,
                 const cudaArray_const_t& buf, std::size_t w, std::size_t h,
                 const spray::geom::point loc,
                 const spray::geom::point lower_left,
                 const spray::geom::point horizontal,
                 const spray::geom::point vertical,
                 const spray::core::world_base& world);

void render(const dim3 blocks, const dim3 threads, const cudaStream_t stream,
            const spray::core::camera& cam, const spray::core::world_base& wld,
            const buffer_array& buf)
{
    render_impl(blocks, threads, stream, buf.array(), buf.width(), buf.height(),
        cam.location(), cam.lower_left(), cam.horizontal(), cam.vertical(), wld);
}

} // cuda
} // spray
#endif// SPRAY_CUDA_RENDER_HPP
