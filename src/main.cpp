#include <spray/glad/load.hpp>
#include <spray/glfw/init.hpp>
#include <spray/glfw/window.hpp>
#include <spray/cuda/buffer_array.hpp>
#include <spray/cuda/render.hpp>

#include <spray/core/camera.hpp>
#include <spray/core/world.hpp>

#include <chrono>

#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_opengl3.h>

int main()
{
    const auto glfw   = spray::glfw::init();
    auto window = spray::glfw::window<std::unique_ptr>(640, 480, "spray");
    spray::glfw::make_context_current(window);

    // load OpenGL function as a function pointer at runtime.
    // this should be called after creating OpenGL context.
    spray::glad::load(glfwGetProcAddress);

    // set vsync interval. this should be called after making context
    spray::glfw::swap_interval(1);

    const auto ini_fbuf_size = spray::glfw::get_frame_buffer_size(window);
    spray::cuda::buffer_array
        bufarray(ini_fbuf_size.first, ini_fbuf_size.second);


    cudaStream_t stream;
    spray::cuda::cuda_assert(
            cudaStreamCreateWithFlags(&stream, cudaStreamDefault));

    spray::core::world          wld;
    spray::core::pinhole_camera cam(
        /* loc = */spray::geom::make_point(0.0, 0.0,  0.0),
        /* dir = */spray::geom::make_point(0.0, 0.0, -1.0),
        /* vup = */spray::geom::make_point(0.0, 1.0,  0.0),
        90.0f,
        640,
        480
        );

    wld.materials.push_back(spray::core::material{spray::core::make_color(1.0, 0.0, 0.0)});
    wld.materials.push_back(spray::core::material{spray::core::make_color(0.0, 1.0, 0.0)});
    wld.materials.push_back(spray::core::material{spray::core::make_color(0.0, 0.0, 1.0)});
    wld.spheres.push_back(spray::geom::make_sphere(spray::geom::make_point( 0.0f, 0.0f, -1.0f), 0.5f));
    wld.spheres.push_back(spray::geom::make_sphere(spray::geom::make_point( 1.0f, 0.0f, -1.0f), 0.5f));
    wld.spheres.push_back(spray::geom::make_sphere(spray::geom::make_point(-1.0f, 0.0f, -1.0f), 0.5f));


    window.set_camera(std::addressof(cam));
    window.set_world (std::addressof(wld));

    spray::glfw::set_key_callback(window,
        [](GLFWwindow* win, int key, int code, int action, int mods) -> void {
            const auto wp = reinterpret_cast<spray::glfw::window_parameter*>(
                    glfwGetWindowUserPointer(win));
            if(!wp->camera)
            {
                return ;
            }
            switch(key)
            {
                case GLFW_KEY_W    : wp->camera->advance( 0.1f); break;
                case GLFW_KEY_S    : wp->camera->advance(-0.1f); break;
                case GLFW_KEY_UP   : wp->camera->pitch  ( 0.01f); break;
                case GLFW_KEY_DOWN : wp->camera->pitch  (-0.01f); break;
                case GLFW_KEY_LEFT : wp->camera->yaw    ( 0.01f); break;
                case GLFW_KEY_RIGHT: wp->camera->yaw    (-0.01f); break;
            }
            return;
        });

    while(!spray::glfw::should_close(window))
    {
        const auto start = std::chrono::system_clock::now();
        const auto size = spray::glfw::get_frame_buffer_size(window);

        const dim3 blocks (size.first / 32, size.second / 32);
        const dim3 threads(32, 32);

        spray::cuda::render(blocks, threads, stream, cam, wld, bufarray);
        spray::cuda::blit_framebuffer(bufarray);

        spray::glfw::swap_buffers(window);

        glfwPollEvents();
        const auto stop = std::chrono::system_clock::now();
        const auto uspf = std::chrono::duration_cast<std::chrono::microseconds>(
                stop - start).count();
        const auto fps  = 1.0e6 / uspf;

        fmt::print(fmt::fg(fmt::color::green), "\rInfo:");
        fmt::print(" {} fps", fps);
    }
    return 0;
}
