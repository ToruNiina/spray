#ifndef SPRAY_CORE_MATERIAL_HPP
#define SPRAY_CORE_MATERIAL_HPP
#include <spray/core/color.hpp>

namespace spray
{
namespace core
{

struct material
{
    spray::core::color albedo;
    spray::core::color emission;
};

} // core
} // spray
#endif// SPRAY_CORE_MATERIAL_HPP
