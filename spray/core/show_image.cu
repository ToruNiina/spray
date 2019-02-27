#include <spray/util/cuda_assert.hpp>
#include <spray/core/color.hpp>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace spray
{
namespace core
{
surface<void, cudaSurfaceType2D> surf_ref;

__global__
void show_image_kernel(const std::size_t width, const std::size_t height,
                       thrust::device_ptr<const spray::core::color> scene)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x >= width || y >= height) {return;}
    const std::size_t offset = x + y * width;

    const uchar4 pixel = spray::core::make_pixel(scene[offset]);
    surf2Dwrite(pixel, surf_ref, x * sizeof(uchar4), y, cudaBoundaryModeZero);
    return;
}

void show_image(const dim3 blocks, const dim3 threads,
        const cudaStream_t stream, const cudaArray_const_t& buf,
        const std::size_t  width,  const std::size_t height,
        thrust::device_ptr<const spray::core::color> scene)
{
    spray::util::cuda_assert(cudaBindSurfaceToArray(surf_ref, buf));
    show_image_kernel<<<blocks, threads, 0, stream>>>(width, height, scene);
    return;
}

} // core
} // spray
