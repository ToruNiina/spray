#ifndef SPRAY_UTIL_CUDA_MACRO_HPP
#define SPRAY_UTIL_CUDA_MACRO_HPP

#if defined(__CUDACC__) && defined(__NVCC__)
#  define SPRAY_HOST        __host__
#  define SPRAY_DEVICE      __device__
#  define SPRAY_HOST_DEVICE __host__ __device__
#  define SPRAY_GLOBAL      __global__
#  define SPRAY_INLINE      __inline__
#else
#  define SPRAY_HOST
#  define SPRAY_DEVICE
#  define SPRAY_HOST_DEVICE
#  define SPRAY_GLOBAL
#  define SPRAY_INLINE      inline
#endif

#endif// SPRAY_UTIL_CUDA_MACRO_HPP
