#ifndef SPRAY_CORE_WORLD_HPP
#define SPRAY_CORE_WORLD_HPP
#include <spray/core/material.hpp>
#include <spray/geom/sphere.hpp>
#include <thrust/host_vector.h>
#include <cstdint>

namespace spray
{
namespace core
{

struct world
{
  public:
    thrust::host_vector<spray::geom::sphere>   spheres;
    thrust::host_vector<spray::core::material> materials;
};

} // core
} // spray
#endif// SPRAY_CORE_WORLD_HPP
