#ifndef SPRAY_CORE_WORLD_BASE_HPP
#define SPRAY_CORE_WORLD_BASE_HPP
#include <spray/core/material.hpp>
#include <spray/geom/sphere.hpp>
#include <memory>

namespace spray
{
namespace core
{

class world_base
{
  public:
    using sphere_type   = spray::geom::sphere;
    using material_type = spray::core::material;

  public:

    virtual ~world_base() = default;

    virtual void push_back(const sphere_type& sph, const material_type& mat) = 0;

    virtual sphere_type const&   sphere_at  (const std::size_t idx) const = 0;
    virtual material_type const& material_at(const std::size_t idx) const = 0;

    virtual void change_sphere_at  (const std::size_t idx, const sphere_type&) = 0;
    virtual void change_material_at(const std::size_t idx, const material_type&) = 0;

    virtual void open_window_for(const std::size_t idx) = 0;
    virtual bool update_gui() = 0;

    virtual bool is_loaded() const noexcept = 0;
    virtual void load() const = 0; // load objects into GPU
};

// some of the codes in STL might use CPU-intrinsics and are not compatible
// with CUDA compiler. To split cuda and c++ source, here it encapsulate the
// world class that may have cuda implementation and returns the interface.
std::unique_ptr<world_base> make_world();

} // core
} // spray
#endif// SPRAY_CORE_WORLD_BASE_HPP
