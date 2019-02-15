#ifndef SPRAY_CUDA_UTIL_CUH
#define SPRAY_CUDA_UTIL_CUH

namespace spray
{
namespace util
{

__device__
float fclampf(float x, float minimum, float maximum)
{
    return fminf(fmaxf(x, minimum), maximum);
}


} // util
} // spray
#endif// SPRAY_CUDA_UTIL_CUH
