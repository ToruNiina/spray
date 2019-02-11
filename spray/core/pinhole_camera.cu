#include <spray/core/pinhole_camera.cuh>
#include <spray/core/render.cuh>
#include <spray/core/show_image.cuh>
#include <spray/core/world.cuh>
/* #include <imgui.h> */

namespace spray
{
namespace core
{

std::unique_ptr<camera_base> make_pinhole_camera(
        std::string        name,
        spray::geom::point location,
        spray::geom::point direction,
        spray::geom::point view_up,
        float              fov,
        std::size_t        width,
        std::size_t        height
    )
{
    return std::make_unique<pinhole_camera>(std::move(name),
            location, direction, view_up, fov, width, height);
}

void pinhole_camera::reset(spray::geom::point location,
                           spray::geom::point direction,
                           spray::geom::point view_up,
                           float              fov,
                           std::size_t        width,
                           std::size_t        height)
{
    this->width_   = width;
    this->height_  = height;
    this->rwidth_  = 1.0f / width;
    this->rheight_ = 1.0f / height;

    const float aspect_ratio = static_cast<float>(width) / height;
    const float theta        = fov * 3.14159265 / 180.0;
    const float half_height  = std::tan(theta * 0.5);
    const float half_width   = half_height * aspect_ratio;

    const auto w = -spray::geom::unit(direction);
    const auto u =  spray::geom::unit(spray::geom::cross(view_up, w));
    const auto v =  spray::geom::cross(w, u);

    this->field_of_view_ = fov;
    this->location_   = location;
    this->direction_  = -w;
    this->view_up_    = v;
    this->pitch_axis_ = u;
    this->lower_left_ = location - half_width * u - half_height * v - w;
    this->horizontal_ = (2 * half_width)  * u;
    this->vertical_   = (2 * half_height) * v;

    if(this->scene_.size() != width * height)
    {
        this->scene_.resize(width * height);
    }

    this->field_of_view_buf_ = this->field_of_view_;
    this->pos_buf_[0] = spray::geom::X(this->location_);
    this->pos_buf_[1] = spray::geom::Y(this->location_);
    this->pos_buf_[2] = spray::geom::Z(this->location_);
    this->dir_buf_[0] = spray::geom::X(this->direction_);
    this->dir_buf_[1] = spray::geom::Y(this->direction_);
    this->dir_buf_[2] = spray::geom::Z(this->direction_);
    this->vup_buf_[0] = spray::geom::X(this->view_up_);
    this->vup_buf_[1] = spray::geom::Y(this->view_up_);
    this->vup_buf_[2] = spray::geom::Z(this->view_up_);
    return ;
}

bool pinhole_camera::update_gui()
{
/*     ImGui::Begin(this->name_.c_str());
    const bool focused = !(ImGui::IsWindowFocused());

    ImGui::InputFloat ("View angle", std::addressof(this->field_of_view_buf_));
    ImGui::InputFloat3("Camera position",  pos_buf_.data());
    ImGui::InputFloat3("Camera direction", dir_buf_.data());
    ImGui::InputFloat3("Camera view-up",   vup_buf_.data());
    if(ImGui::Button("Apply changes"))
    {
        this->reset(
            spray::geom::make_point(pos_buf_[0], pos_buf_[1], pos_buf_[2]),
            spray::geom::make_point(dir_buf_[0], dir_buf_[1], dir_buf_[2]),
            spray::geom::make_point(vup_buf_[0], vup_buf_[1], vup_buf_[2]),
            this->field_of_view_buf_, this->width_, this->height_);
    }

//        ImGui::InputText("File name", filename_buf_.data(), 256);
//        if(ImGui::Button("Save image as png"))
//        {
//        }

    ImGui::End();
    return focused; */
    return false;
}

void pinhole_camera::render(
        const dim3 blocks, const dim3 threads, const cudaStream_t stream,
        const spray::core::world_base& wld_base,
        const spray::cuda::buffer_array& bufarray)
{
    const auto& wld = dynamic_cast<spray::core::world const&>(wld_base);
    if(!wld.is_loaded())
    {
        wld.load();
    }

    spray::cuda::render_kernel<<<blocks, threads, 0, stream>>>(
        this->width_, this->height_, this->rwidth_, this->rheight_,
        this->location_, this->lower_left_, this->horizontal_, this->vertical_,
        wld.device_spheres().size(),
        thrust::device_pointer_cast(wld.device_materials().data()),
        thrust::device_pointer_cast(wld.device_spheres().data()),
        thrust::device_pointer_cast(this->scene_.data())
        );

    spray::cuda::show_image(
           blocks, threads, stream, bufarray.array(), this->width_, this->height_,
           thrust::device_pointer_cast(this->scene_.data()));
    return;
}





} // core
} // spray
