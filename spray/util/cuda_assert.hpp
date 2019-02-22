#ifndef SPRAY_UTIL_CUDA_ASSERT_HPP
#define SPRAY_UTIL_CUDA_ASSERT_HPP
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
        std::string description("Error(");
        description += std::string(cudaGetErrorName(err));
        description += "): ";
        description += std::string(cudaGetErrorString(err));
        throw std::runtime_error(std::move(description));
    }
    return;
}

} // util
} // spray
#endif// SPRAY_UTIL_CUDA_ASSERT_HPP
