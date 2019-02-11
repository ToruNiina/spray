#ifndef SPRAY_CORE_WORLD_CUH
#define SPRAY_CORE_WORLD_CUH
#include <spray/core/world_base.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <string>

namespace spray
{
namespace core
{

class world : public world_base
{
  public:

    using sphere_type   = typename world_base::sphere_type;
    using material_type = typename world_base::material_type;

  public:

    world() = default;
    ~world() override = default;

    sphere_type const&   sphere_at  (const std::size_t idx) const override
    {
        if(host_spheres_.size() <= idx)
        {
            throw std::out_of_range("spray::core::world::sphere_at(): idx (" +
                    std::to_string(idx) + std::string(") exceeds size (") +
                    std::to_string(host_spheres_.size()) + std::string(")."));
        }
        return host_spheres_[idx];
    }
    material_type const& material_at(const std::size_t idx) const override
    {
        if(host_materials_.size() <= idx)
        {
            throw std::out_of_range("spray::core::world::material_at(): idx (" +
                    std::to_string(idx) + std::string(") exceeds size (") +
                    std::to_string(host_materials_.size()) + std::string(")."));
        }
        return host_materials_[idx];
    }

    void change_sphere_at  (const std::size_t idx, const sphere_type& new_sphere) override
    {
        if(host_spheres_.size() <= idx)
        {
            throw std::out_of_range("spray::core::world::change_sphere_at(): idx (" +
                    std::to_string(idx) + std::string(") exceeds size (") +
                    std::to_string(host_spheres_.size()) + std::string(")."));
        }
        is_loaded_ = false;
        host_spheres_[idx] = new_sphere;
        return;
    }
    void change_material_at(const std::size_t idx, const material_type& new_material) override
    {
        if(host_materials_.size() <= idx)
        {
            throw std::out_of_range("spray::core::world::change_material_at(): idx (" +
                    std::to_string(idx) + std::string(") exceeds size (") +
                    std::to_string(host_materials_.size()) + std::string(")."));
        }
        is_loaded_ = false;
        host_materials_[idx] = new_material;
        return;
    }

    void push_back(const sphere_type& sph, const material_type& mat) override
    {
        is_loaded_ = false;
        host_spheres_.push_back(sph);
        host_materials_.push_back(mat);
        return;
    }

    bool is_loaded() const noexcept {return this->is_loaded_;}

    void load() const
    {
        device_spheres_   = host_spheres_;
        device_materials_ = host_materials_;
    }

    std::size_t first_hit_object(std::size_t w, std::size_t h) const
    {
        return 0;
    }

    thrust::device_vector<sphere_type> const&
    device_spheres() const noexcept {return this->device_spheres_;}

    thrust::device_vector<material_type> const&
    device_materials() const noexcept {return this->device_materials_;}

  private:

    bool is_loaded_;
    thrust::host_vector<sphere_type>     host_spheres_;
    thrust::host_vector<material_type>   host_materials_;
    mutable thrust::device_vector<sphere_type>   device_spheres_;
    mutable thrust::device_vector<material_type> device_materials_;
};

} // core
} // spray
#endif// SPRAY_CORE_WORLD_CUH
