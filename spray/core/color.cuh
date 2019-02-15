#ifndef SPRAY_CORE_COLOR_CUH
#define SPRAY_CORE_COLOR_CUH
#include <spray/core/color.hpp>
#include <spray/util/cuda_util.cuh>

namespace spray
{
namespace core
{
    
__device__
uchar4 make_pixel(spray::core::color col)
{
    uchar4 pixel;
    pixel.x = std::uint8_t(spray::util::fclampf(sqrtf(spray::core::R(col)) * 256, 0, 255));
    pixel.y = std::uint8_t(spray::util::fclampf(sqrtf(spray::core::G(col)) * 256, 0, 255));
    pixel.z = std::uint8_t(spray::util::fclampf(sqrtf(spray::core::B(col)) * 256, 0, 255));
    pixel.w = 0xFF;
    return pixel;
}

} // core
} // spray
#endif // SPRAY_CORE_COLOR_CUH
