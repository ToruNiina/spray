#ifndef SPRAY_CUDA_SHOW_IMAGE_HPP
#define SPRAY_CUDA_SHOW_IMAGE_HPP
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <vector_types.h>

namespace spray
{
namespace cuda
{

void show_image(const dim3 blocks, const dim3 threads,
        const cudaStream_t stream, const cudaArray_const_t& buf,
        const std::size_t  width,  const std::size_t height,
        thrust::device_ptr<const uchar4> image);

void load_image(const dim3 blocks, const dim3 threads,
        const cudaStream_t stream, const cudaArray_const_t& buf,
        const std::size_t  width,  const std::size_t height,
        const thrust::host_vector<uchar4>& image);

} // cuda
} // spray
#endif // SPRAY_CUDA_SHOW_IMAGE_HPP
