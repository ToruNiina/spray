#ifndef SPRAY_UTIL_CUDA_MATH_HPP
#define SPRAY_UTIL_CUDA_MATH_HPP
#include <spray/util/cuda_macro.hpp>
#include <math_constants.h>
#include <limits>
#include <cmath>

#ifndef __CUDACC__
#include <algorithm>
#endif

namespace spray
{
namespace util
{

SPRAY_HOST_DEVICE
SPRAY_INLINE float fclampf(float x, float minimum, float maximum) noexcept
{
#ifdef __CUDA_ARCH__
    return fminf(fmaxf(x, minimum), maximum);
#else
    return std::min(std::max(x, minimum), maximum);
#endif
}


SPRAY_HOST_DEVICE
SPRAY_INLINE float inf() noexcept
{
#ifdef __CUDA_ARCH__
    return CUDART_INF_F;
#else
    return std::numeric_limits<float>::infinity();
#endif
}

SPRAY_HOST_DEVICE
constexpr inline float epsilon() noexcept
{
    return 1.192e-7;
}


} // util
} // spray
#endif// SPRAY_UTIL_CUDA_MATH_HPP
