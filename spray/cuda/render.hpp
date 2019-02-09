#ifndef SPRAY_CUDA_RENDER_HPP
#define SPRAY_CUDA_RENDER_HPP
#include <spray/core/world.hpp>
#include <spray/core/camera.hpp>
#include <spray/cuda/buffer_array.hpp>

namespace spray
{
namespace cuda
{

void render_impl(const dim3 blocks, const dim3 threads, const cudaStream_t stream,
                 const cudaArray_const_t& buf, std::size_t w, std::size_t h);

void render(const dim3 blocks, const dim3 threads, const cudaStream_t stream,
            const spray::core::camera& cam, const spray::core::world& wld,
            const buffer_array& buf)
{
    render_impl(blocks, threads, stream, buf.array(), buf.width(), buf.height());
}

} // cuda
} // spray
#endif// SPRAY_CUDA_RENDER_HPP
