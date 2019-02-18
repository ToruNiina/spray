#ifndef SPRAY_CORE_SAVE_IMAGE_HPP
#define SPRAY_CORE_SAVE_IMAGE_HPP
#include <thrust/host_vector.h>
#include <vector_type.h>

namespace spray
{
namespace core
{

void save_image(const std::size_t w, const std::size_t h,
                const thrust::host_vector<uchar4>& pixels,
                const char* filename);

} // core
} // spray
#endif// SPRAY_CORE_SAVE_IMAGE_HPP
