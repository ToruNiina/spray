#ifndef SPRAY_CUDA_CUDA_ASSERT_HPP
#define SPRAY_CUDA_CUDA_ASSERT_HPP
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace spray
{
namespace core
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

} // core
} // spray
#endif// SPRAY_CUDA_CUDA_ASSERT_HPP
