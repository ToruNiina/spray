#ifndef SPRAY_CORE_CAMERA_BASE_HPP
#define SPRAY_CORE_CAMERA_BASE_HPP
#include <spray/core/world_base.hpp>
#include <spray/core/buffer_array.hpp>
#include <spray/geom/point.hpp>

namespace spray
{
namespace core
{

struct camera_base
{
    virtual std::string const& name() const noexcept = 0;
    virtual std::size_t width()  const noexcept = 0;
    virtual std::size_t height() const noexcept = 0;
    virtual spray::geom::point location()   const noexcept = 0;
    virtual spray::geom::point direction()  const noexcept = 0;
    virtual spray::geom::point lower_left() const noexcept = 0;
    virtual spray::geom::point horizontal() const noexcept = 0;
    virtual spray::geom::point vertical()   const noexcept = 0;
    virtual float focus_dist() const = 0;

    // get object index corresponds to the pixel index.
    virtual std::size_t first_hit_object(std::size_t w, std::size_t h) const = 0;

    virtual void focus_dist(float dist) = 0;
    virtual void resize (std::size_t w, std::size_t h) = 0;
    virtual void move   (spray::geom::point position)  = 0;
    virtual void look   (spray::geom::point direction) = 0;
    virtual void advance(float dist)  = 0;
    virtual void lateral(float dist)  = 0;
    virtual void yaw    (float angle) = 0;
    virtual void pitch  (float angle) = 0;
    virtual void roll   (float angle) = 0;

    virtual bool update_gui() = 0; // returns true if the window is focused
    virtual void render(const cudaStream_t, const spray::core::world_base&,
                        const buffer_array&) = 0;
};

std::unique_ptr<camera_base> make_pinhole_camera(
        std::string        name,
        spray::geom::point location,
        spray::geom::point direction,
        spray::geom::point view_up,
        float              fov,
        std::size_t        width,
        std::size_t        height
    );

std::unique_ptr<camera_base> make_orthogonal_camera(
        std::string        name,
        spray::geom::point location,
        spray::geom::point direction,
        spray::geom::point view_up,
        float              fov,
        std::size_t        width,
        std::size_t        height
    );

} // core
} // spray
#endif// SPRAY_CORE_CAMERA_HPP
