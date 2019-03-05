#include <spray/core/pinhole_camera.cuh>
#include <spray/core/path_tracing.cuh>
#include <spray/core/show_image.cuh>
#include <spray/core/save_image.hpp>
#include <spray/core/world.cuh>
#include <spray/geom/ray.hpp>
#include <spray/geom/collide.hpp>
#include <spray/util/cuda_assert.hpp>
#include <spray/util/log.hpp>

#include <thrust/tuple.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <imgui.h>

namespace spray
{
namespace core
{

__global__
void render_kernel(const std::uint32_t weight,
        const std::size_t width, const std::size_t height,
        const float      rwidth, const float      rheight,
        const spray::geom::point location,
        const spray::geom::point lower_left,
        const spray::geom::point horizontal,
        const spray::geom::point vertical,
        const std::size_t        N,
        const spray::core::color background,
        thrust::device_ptr<const spray::core::material> material,
        thrust::device_ptr<const spray::geom::sphere>   spheres,
        thrust::device_ptr<spray::core::color> img,
        thrust::device_ptr<std::uint32_t>      first_hit_obj,
        thrust::device_ptr<std::uint32_t>      seeds)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x >= width || y >= height) {return;}

    const std::size_t offset = x + y * width;
    thrust::default_random_engine rng(seeds[offset]);
    thrust::uniform_real_distribution<float> uni(0.0f, 1.0f);

    const spray::geom::point dst = lower_left +
                                   ((float(x) + uni(rng)) *  rwidth) * horizontal +
                                   ((float(y) + uni(rng)) * rheight) * vertical;
    const spray::geom::ray ray(location, dst - location);

    const auto pix_idx = path_trace(ray, background, 16, N, material, spheres, rng);

    img[offset] = (img[offset] * weight + pix_idx.first) / (weight + 1);
    first_hit_obj[offset] = pix_idx.second;
    seeds[offset]         = rng();
    return;
}

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
    this->weight_  = 0u;
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
        this->host_first_hit_obj_.resize(width * height);
        this->device_first_hit_obj_.resize(width * height);

        this->device_seeds_.resize(width * height);
        thrust::transform(
                thrust::counting_iterator<std::uint32_t>(0),
                thrust::counting_iterator<std::uint32_t>(width * height),
                this->device_seeds_.begin(),
                [] __device__ (const std::uint32_t n) {
                    thrust::default_random_engine rng;
                    rng.discard(n);
                    return rng();
                });
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
    ImGui::Begin(this->name_.c_str());
    const bool focused = ImGui::IsWindowFocused();

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

    ImGui::InputText("File name", filename_buf_.data(), 256);
    if(ImGui::Button("Save image as png"))
    {
        const thrust::host_vector<spray::core::color> scene = this->scene_;
        std::vector<uchar4> pixels; pixels.reserve(scene.size());
        for(const auto& col : scene)
        {
            pixels.push_back(spray::core::make_pixel(col));
        }
        assert(pixels.size() == scene.size());

        save_image(this->width_, this->height_, pixels,
                   this->filename_buf_.data());
    }

    ImGui::Text("Currently %d frames accumurated", this->weight_);
    const auto framerate = ImGui::GetIO().Framerate;
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / framerate, framerate);
    ImGui::End();
    return focused;
}

void pinhole_camera::render(const cudaStream_t stream,
        const world_base& wld_base, const buffer_array& bufarray)
{
    const auto& wld = dynamic_cast<spray::core::world const&>(wld_base);
    if(!wld.is_loaded())
    {
        wld.load();
    }

    cudaFuncAttributes attr;
    spray::util::cuda_assert(cudaFuncGetAttributes(&attr, render_kernel));
//     spray::log(spray::log_level::debug,
//                "max number of threads per block allowed for the kernel is ",
//                attr.maxThreadsPerBlock, '\n');

    cudaDeviceProp prop;
    spray::util::cuda_assert(cudaGetDeviceProperties(&prop, 0));

    const auto warp_number = attr.maxThreadsPerBlock / prop.warpSize;
    const dim3 threads(prop.warpSize, warp_number);
    const dim3 blocks(std::ceil(double(bufarray.width())  / threads.x),
                      std::ceil(double(bufarray.height()) / threads.y));

    render_kernel<<<blocks, threads, 0, stream>>>(this->weight_,
        this->width_, this->height_, this->rwidth_, this->rheight_,
        this->location_, this->lower_left_, this->horizontal_, this->vertical_,
        wld.device_spheres().size(), wld.background(),
        thrust::device_pointer_cast(wld.device_materials().data()),
        thrust::device_pointer_cast(wld.device_spheres().data()),
        thrust::device_pointer_cast(this->scene_.data()),
        thrust::device_pointer_cast(this->device_first_hit_obj_.data()),
        thrust::device_pointer_cast(this->device_seeds_.data())
        );
    spray::util::cuda_assert(cudaPeekAtLastError());

    this->host_first_hit_obj_ = this->device_first_hit_obj_;

    spray::core::show_image(
       blocks, threads, stream, bufarray.array(), this->width_, this->height_,
       thrust::device_pointer_cast(this->scene_.data()));
    spray::util::cuda_assert(cudaPeekAtLastError());

    this->weight_ = (this->weight_ == 0xFFFFFFFE) ? 0xFFFFFFFE : this->weight_+1;
    return;
}

} // core
} // spray
