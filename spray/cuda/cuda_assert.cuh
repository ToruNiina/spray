#ifndef SPRAY_CUDA_CUDA_ASSERT_HPP
#define SPRAY_CUDA_CUDA_ASSERT_HPP
#include <spray/util/log.hpp>
#include <cuda_runtime.h>

namespace spray
{
namespace cuda
{

inline void cuda_assert(cudaError_t err)
{
    if(err != cudaSuccess)
    {
        spray::log(spray::log_level::error, "{} ({}) occured.",
                   cudaGetErrorName(err), cudaGetErrorString(err));
        throw std::runtime_error(fmt::format("Error: {} ({})",
                cudaGetErrorName(err), cudaGetErrorString(err)));
    }
    return;
}

} // cuda
} // spray
#endif// SPRAY_CUDA_CUDA_ASSERT_HPP
