#include <spray/core/cuda_assert.hpp>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace spray
{
namespace cuda
{
surface<void, cudaSurfaceType2D> surf_ref;

__global__
void show_image_kernel(const std::size_t width, const std::size_t height,
                       thrust::device_ptr<const uchar4> image)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x >= width)  {return;}
    if(y >= height) {return;}
    const std::size_t offset = x + y * width;

    const uchar4 pixel = image[offset];
    surf2Dwrite(pixel, surf_ref, x * sizeof(uchar4), y, cudaBoundaryModeZero);
    return;
}

void show_image(const dim3 blocks, const dim3 threads,
        const cudaStream_t stream, const cudaArray_const_t& buf,
        const std::size_t  width,  const std::size_t height,
        thrust::device_ptr<const uchar4> image)
{
    cuda_assert(cudaBindSurfaceToArray(surf_ref, buf));
    show_image_kernel<<<blocks, threads, 0, stream>>>(width, height, image);
    return;
}

void load_image(const dim3 blocks, const dim3 threads,
        const cudaStream_t stream, const cudaArray_const_t& buf,
        const std::size_t  width,  const std::size_t height,
        const thrust::host_vector<uchar4>& image)
{
    const thrust::device_vector<uchar4> image_device = image;
    show_image(blocks, threads, stream, buf, width, height,
               thrust::device_pointer_cast(image_device.data()));
    return;
}



} // cuda
} // spray
