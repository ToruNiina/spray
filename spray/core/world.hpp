#ifndef SPRAY_CORE_WORLD_HPP
#define SPRAY_CORE_WORLD_HPP
#include <thrust/device_vector.h>
#include <spray/geom/sphere.hpp>

namespace spray
{
namespace core
{

struct world
{
    thrust::device_vector<spray::geom::sphere> spheres_;
};

} // core
} // spray
#endif// SPRAY_CORE_WORLD_HPP
