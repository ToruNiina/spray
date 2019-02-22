#ifndef SPRAY_CUDA_UTIL_HPP
#define SPRAY_CUDA_UTIL_HPP
#include <spray/core/cuda_macro.hpp>
#include <cmath>

namespace spray
{
namespace util
{

SPRAY_HOST_DEVICE
SPRAY_INLINE float fclampf(float x, float minimum, float maximum)
{
#ifdef __CUDA_ARCH__
    return fminf(fmaxf(x, minimum), maximum);
#else
    return std::min(std::max(x, minimum), maximum);
#endif
}


} // util
} // spray
#endif// SPRAY_CUDA_UTIL_HPP
