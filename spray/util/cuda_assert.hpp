#ifndef SPRAY_UTIL_CUDA_ASSERT_HPP
#define SPRAY_UTIL_CUDA_ASSERT_HPP
#include <spray/util/log.hpp>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace spray
{
namespace util
{

inline void cuda_assert(cudaError_t err)
{
    if(err != cudaSuccess)
    {
        spray::log(spray::log_level::error, cudaGetErrorName(err), ": ",
                                            cudaGetErrorString(err), '\n');
        throw std::runtime_error(std::string(cudaGetErrorName(err)) +
            std::string(": ") + std::string(cudaGetErrorString(err)));
    }
    return;
}

} // util
} // spray
#endif// SPRAY_UTIL_CUDA_ASSERT_HPP
