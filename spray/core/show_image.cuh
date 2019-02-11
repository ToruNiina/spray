#ifndef SPRAY_CORE_SHOW_IMAGE_CUH
#define SPRAY_CORE_SHOW_IMAGE_CUH
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <vector_types.h>

namespace spray
{
namespace core
{

void show_image(const dim3 blocks, const dim3 threads,
        const cudaStream_t stream, const cudaArray_const_t& buf,
        const std::size_t  width,  const std::size_t height,
        thrust::device_ptr<const uchar4> image);

void load_image(const dim3 blocks, const dim3 threads,
        const cudaStream_t stream, const cudaArray_const_t& buf,
        const std::size_t  width,  const std::size_t height,
        const thrust::host_vector<uchar4>& image);

} // core
} // spray
#endif // SPRAY_CORE_SHOW_IMAGE_CUH
