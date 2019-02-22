#include <png++/png.hpp>
#include <spray/core/save_image.hpp>

namespace spray
{
namespace core
{

void save_image(const std::size_t w, const std::size_t h,
                const thrust::host_vector<uchar4>& pixels,
                const char* filename)
{
    assert(pixels.size() == w * h);

    png::image<png::rgba_pixel> img(w, h);
    for(std::size_t y=0; y<h; ++y)
    {
        for(std::size_t x=0; x<w; ++x)
        {
            const auto pix = pixels[x + y * w];
            img[y][x] = png::rgba_pixel(pix.x, pix.y, pix.z, pix.w);
        }
    }
    img.write(filename);
    return;
}

} // core
} // spray
