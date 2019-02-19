#ifndef SPRAY_CORE_ORTHOGONAL_CAMERA_CUH
#define SPRAY_CORE_ORTHOGONAL_CAMERA_CUH
#include <spray/core/camera_base.hpp>
#include <spray/core/world_base.hpp>
#include <thrust/device_vector.h>

namespace spray
{
namespace core
{

struct orthogonal_camera final : public camera_base
{
  public:
    orthogonal_camera(std::string        name,
                      spray::geom::point location,
                      spray::geom::point direction,
                      spray::geom::point view_up,
                      float              fov,
                      std::size_t        width,
                      std::size_t        height)
        : name_(std::move(name))
    {
        this->reset(location, direction, view_up, fov, width, height);
    }

    void reset(spray::geom::point location,
               spray::geom::point direction,
               spray::geom::point view_up,
               float              fov,
               std::size_t        width,
               std::size_t        height);

    bool update_gui() override;

    void render(const dim3 blocks, const dim3 threads, const cudaStream_t stream,
            const world_base& wld_base, const buffer_array& bufarray) override;

    std::string const& name() const noexcept {return this->name_;}
    std::size_t width()  const noexcept override {return width_;}
    std::size_t height() const noexcept override {return height_;}
    spray::geom::point location()   const noexcept override {return location_;}
    spray::geom::point direction()  const noexcept override {return direction_;}
    spray::geom::point lower_left() const noexcept override {return lower_left_;}
    spray::geom::point horizontal() const noexcept override {return horizontal_;}
    spray::geom::point vertical()   const noexcept override {return vertical_;}

    std::size_t first_hit_object(std::size_t w, std::size_t h) const
    {
        if(this->width_ <= w || this->height_ <= h)
        {
            throw std::out_of_range("orthogonal_camera::first_hit_object");
        }
        return this->host_first_hit_obj_[w + h * this->width_];
    }

    void  focus_dist(float) override {} // do nothing
    float focus_dist() const override {return 0.0;}

    void resize(std::size_t w, std::size_t h) override
    {
        this->reset(this->location_,
                    this->direction_,
                    this->view_up_,
                    this->field_of_view_,
                    w, h);
    }


    void look(spray::geom::point new_direction) override
    {
        this->reset(this->location_,
                    new_direction,
                    this->view_up_,
                    this->field_of_view_,
                    this->width_,
                    this->height_);
    }
    void move(spray::geom::point new_position) override
    {
        this->reset(new_position,
                    this->direction_,
                    this->view_up_,
                    this->field_of_view_,
                    this->width_,
                    this->height_);
    }
    void advance(float dist) override
    {
        this->reset(this->location_ + dist * direction_,
                    this->direction_,
                    this->view_up_,
                    this->field_of_view_,
                    this->width_,
                    this->height_);
    }
    void lateral(float dist) override
    {
        this->reset(this->location_ + dist * pitch_axis_,
                    this->direction_,
                    this->view_up_,
                    this->field_of_view_,
                    this->width_,
                    this->height_);
    }
    void yaw  (float angle) override
    {
        this->reset(this->location_,
                    spray::geom::rotate(this->direction_, -angle, this->view_up_),
                    this->view_up_,
                    this->field_of_view_,
                    this->width_,
                    this->height_);
    }
    void pitch(float angle) override
    {
        this->reset(this->location_,
                    spray::geom::rotate(this->direction_, angle, this->pitch_axis_),
                    this->view_up_,
                    this->field_of_view_,
                    this->width_,
                    this->height_);
    }
    void roll (float angle) override
    {
        this->reset(this->location_,
                    this->direction_,
                    spray::geom::rotate(this->view_up_, angle, this->direction_),
                    this->field_of_view_,
                    this->width_,
                    this->height_);
    }

  private:
    float field_of_view_;
    float rwidth_;
    float rheight_;
    std::size_t width_;
    std::size_t height_;
    spray::geom::point view_up_;
    spray::geom::point direction_;
    spray::geom::point pitch_axis_;

    spray::geom::point location_;
    spray::geom::point lower_left_;
    spray::geom::point horizontal_;
    spray::geom::point vertical_;

    // gui stuff
    float field_of_view_buf_;
    std::array<float, 3> pos_buf_;
    std::array<float, 3> dir_buf_;
    std::array<float, 3> vup_buf_;
    std::array<char, 256> filename_buf_;
    std::string name_;

    // scene image
    thrust::device_vector<uchar4> scene_;

    thrust::host_vector<std::uint32_t>   host_first_hit_obj_;
    thrust::device_vector<std::uint32_t> device_first_hit_obj_;

};

} // core
} // spray
#endif// SPRAY_CORE_orthogonal_CAMERA_CUH
