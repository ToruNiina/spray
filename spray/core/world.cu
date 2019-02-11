#include <spray/core/world.cuh>

namespace spray
{
namespace core
{

std::unique_ptr<world_base> make_world()
{
    return std::make_unique<world>();
}

} // core
} // spray
