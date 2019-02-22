#include <spray/core/orthogonal_camera.cuh>
#include <spray/core/show_image.cuh>
#include <spray/core/save_image.hpp>
#include <spray/core/world.cuh>
#include <spray/geom/ray.hpp>
#include <spray/geom/collide.hpp>
#include <imgui.h>

namespace spray
{
namespace core
{

__global__
void render_orthogonal_kernel(
        const std::size_t width, const std::size_t height,
        const float      rwidth, const float      rheight,
        const spray::geom::point direction,
        const spray::geom::point lower_left,
        const spray::geom::point horizontal,
        const spray::geom::point vertical,
        const std::size_t        N,
        thrust::device_ptr<const spray::core::material> material,
        thrust::device_ptr<const spray::geom::sphere>   spheres,
        thrust::device_ptr<uchar4> img,
        thrust::device_ptr<std::uint32_t> first_hit
        );



std::unique_ptr<camera_base> make_orthogonal_camera(
        std::string        name,
        spray::geom::point location,
        spray::geom::point direction,
        spray::geom::point view_up,
        float              fov,
        std::size_t        width,
        std::size_t        height
    )
{
    return std::make_unique<orthogonal_camera>(std::move(name),
            location, direction, view_up, fov, width, height);
}

void orthogonal_camera::reset(spray::geom::point location,
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
    const float half_height  = fov * 0.5f;
    const float half_width   = half_height * aspect_ratio;

    const auto w = -spray::geom::unit(direction);
    const auto u =  spray::geom::unit(spray::geom::cross(view_up, w));
    const auto v =  spray::geom::cross(w, u);

    this->field_of_view_ = fov;
    this->location_      = location;
    this->direction_     = -w;
    this->view_up_       = v;
    this->pitch_axis_    = u;
    this->lower_left_    = location - half_width * u - half_height * v;
    this->horizontal_    = (2.0f * half_width)  * u;
    this->vertical_      = (2.0f * half_height) * v;

    if(this->scene_.size() != width * height)
    {
        this->scene_.resize(width * height);
        this->host_first_hit_obj_.resize(width * height);
        this->device_first_hit_obj_.resize(width * height);
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

bool orthogonal_camera::update_gui()
{
    ImGui::Begin(this->name_.c_str());
    const bool focused = ImGui::IsWindowFocused();

    ImGui::InputFloat ("View range", std::addressof(this->field_of_view_buf_));
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

    ImGui::InputText("File name", filename_buf_.data(), 256);
    if(ImGui::Button("Save image as png"))
    {
        const thrust::host_vector<uchar4> pixels = this->scene_;
        save_image(this->width_, this->height_, pixels,
                   this->filename_buf_.data());
    }

    const auto framerate = ImGui::GetIO().Framerate;
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / framerate, framerate);
    ImGui::End();
    return focused;
}

void orthogonal_camera::render(
        const dim3 blocks, const dim3 threads, const cudaStream_t stream,
        const world_base& wld_base, const buffer_array& bufarray)
{
    const auto& wld = dynamic_cast<spray::core::world const&>(wld_base);
    if(!wld.is_loaded())
    {
        wld.load();
    }

    spray::core::render_orthogonal_kernel<<<blocks, threads, 0, stream>>>(
        this->width_, this->height_, this->rwidth_, this->rheight_,
        this->direction_, this->lower_left_, this->horizontal_, this->vertical_,
        wld.device_spheres().size(),
        thrust::device_pointer_cast(wld.device_materials().data()),
        thrust::device_pointer_cast(wld.device_spheres().data()),
        thrust::device_pointer_cast(this->scene_.data()),
        thrust::device_pointer_cast(this->device_first_hit_obj_.data())
        );
    this->host_first_hit_obj_ = device_first_hit_obj_;

    spray::core::show_image(
           blocks, threads, stream, bufarray.array(), this->width_, this->height_,
           thrust::device_pointer_cast(this->scene_.data()));
    return;
}

__global__
void render_orthogonal_kernel(
        const std::size_t width, const std::size_t height,
        const float      rwidth, const float      rheight,
        const spray::geom::point direction,
        const spray::geom::point lower_left,
        const spray::geom::point horizontal,
        const spray::geom::point vertical,
        const std::size_t        N,
        thrust::device_ptr<const spray::core::material> material,
        thrust::device_ptr<const spray::geom::sphere>   spheres,
        thrust::device_ptr<uchar4> img,
        thrust::device_ptr<std::uint32_t> first_hit_obj)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x >= width || y >= height) {return;}

    const std::size_t offset = x + y * width;

    const spray::geom::point src = lower_left +
                                   ((x+0.5f) *  rwidth) * horizontal +
                                   ((y+0.5f) * rheight) * vertical;
    const spray::geom::ray ray = spray::geom::make_ray(src, direction);

    std::uint32_t index = 0xFFFFFFFF;
    spray::geom::collision col;
    col.t = spray::geom::inf();
    for(std::size_t i=0; i<N; ++i)
    {
        const spray::geom::collision c = collide(ray, spheres[i], 0.0f);
        if(!isinf(c.t) && c.t < col.t)
        {
            index = i;
            col   = c;
        }
    }

    uchar4 pixel;
    if(index == 0xFFFFFFFF)
    {
        pixel.x = 0x00;
        pixel.y = 0x00;
        pixel.z = 0x00;
        pixel.w = 0x00;
    }
    else
    {
        const spray::core::material mat = material[index];
        const spray::core::color  color = mat.albedo * fabsf(spray::geom::dot(
                spray::geom::direction(ray), spray::geom::normal(col)));

        pixel = make_pixel(color);
    }
    img[offset] = pixel;
    first_hit_obj[offset] = index;
    return;
}

} // core
} // spray
