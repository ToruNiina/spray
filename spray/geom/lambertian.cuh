#ifndef SPRAY_GEOM_LAMBERTIAN_CUH
#define SPRAY_GEOM_LAMBERTIAN_CUH
#include <thrust/random.h>

namespace spray
{
namespace geom
{

template<typename RNG>
__device__ __inline__
point pick_on_sphere(RNG& rng)
{
    thrust::random::normal_distribution<float> nrm(0.0f, 1.0f);
    const float x = nrm(rng);
    const float y = nrm(rng);
    const float z = nrm(rng);
    const float l = x*x + y*y + z*z;
    const float r = rsqrt(l);
    return make_point(x*r, y*r, z*r);
}

template<typename RNG>
__device__ __inline__
ray scatter_lambertian(ray r, point c, point n, RNG& rng)
{
    return make_ray(c, n + pick_on_sphere(rng));
}

} // geom
} // spray
#endif// SPRAY_GEOM_LAMBERTIAN_HPP