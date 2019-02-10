#ifndef SPRAY_CORE_CAMERA_HPP
#define SPRAY_CORE_CAMERA_HPP
#include <spray/geom/point.hpp>

namespace spray
{
namespace core
{

struct camera
{
    virtual std::size_t width()  const noexcept = 0;
    virtual std::size_t height() const noexcept = 0;
    virtual spray::geom::point location()   const noexcept = 0;
    virtual spray::geom::point lower_left() const noexcept = 0;
    virtual spray::geom::point horizontal() const noexcept = 0;
    virtual spray::geom::point vertical()   const noexcept = 0;
};

struct pinhole_camera : public camera
{
  public:
    pinhole_camera(spray::geom::point location,
                   spray::geom::point direction,
                   spray::geom::point view_up,
                   float              fov,
                   std::size_t        width,
                   std::size_t        height)
        : rwidth_(1.0f / width), rheight_(1.0f / height),
          width_(width), height_(height), location_(location)
    {
        const float aspect_ratio = static_cast<float>(width) / height;
        const float theta        = fov * 3.14159265 / 180.0;
        const float half_height  = std::tan(theta * 0.5);
        const float half_width   = half_height * aspect_ratio;

        const auto w = -spray::geom::unit(direction);
        const auto u =  spray::geom::unit(spray::geom::cross(view_up, w));
        const auto v =  spray::geom::cross(w, u);

        this->lower_left_ = location - half_width * u - half_height * v - w;
        this->horizontal_ = (2 * half_width)  * u;
        this->vertical_   = (2 * half_height) * v;
    }

    std::size_t width()  const noexcept override {return width_;}
    std::size_t height() const noexcept override {return height_;}
    spray::geom::point location()   const noexcept override {return location_;}
    spray::geom::point lower_left() const noexcept override {return lower_left_;}
    spray::geom::point horizontal() const noexcept override {return horizontal_;}
    spray::geom::point vertical()   const noexcept override {return vertical_;}

  private:
    float rwidth_;
    float rheight_;
    std::size_t width_;
    std::size_t height_;
    spray::geom::point location_;
    spray::geom::point lower_left_;
    spray::geom::point horizontal_;
    spray::geom::point vertical_;
};

} // core
} // spray
#endif// SPRAY_CORE_CAMERA_HPP
