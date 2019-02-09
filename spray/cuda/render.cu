#include <spray/cuda/cuda_assert.hpp>

namespace spray
{
namespace cuda
{

surface<void, cudaSurfaceType2D> surf_ref;

__device__
float fclampf(float x, float minimum, float maximum)
{
    return fminf(fmaxf(x, minimum), maximum);
}

__global__
void render_kernel(const std::size_t width, const std::size_t height)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const std::size_t offset = x + y * blockDim.x * gridDim.x;
    if(offset >= width * height)
    {
        return;
    }

    const float r = static_cast<float>(x) / static_cast<float>(width);
    const float g = static_cast<float>(y) / static_cast<float>(height);
    const float b = 0.2;

    uchar4 pixel;
    pixel.x = static_cast<unsigned char>(fclampf(r * 256, 0, 255));
    pixel.y = static_cast<unsigned char>(fclampf(g * 256, 0, 255));
    pixel.z = static_cast<unsigned char>(fclampf(b * 256, 0, 255));
    pixel.w = 0xFF;

    surf2Dwrite(pixel, surf_ref, x * sizeof(uchar4), y, cudaBoundaryModeZero);
    return;
}

void render_impl(const dim3 blocks, const dim3 threads, const cudaStream_t stream,
                 const cudaArray_const_t& buf, std::size_t w, std::size_t h)
{
    cuda_assert(cudaBindSurfaceToArray(surf_ref, buf));

    render_kernel<<<blocks, threads, 0, stream>>>(w, h);
    return;
}

} // cuda
} // spray
