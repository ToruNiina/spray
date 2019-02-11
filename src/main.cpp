#include <spray/glad/load.hpp>
#include <spray/glfw/init.hpp>
#include <spray/glfw/window.hpp>
#include <spray/core/buffer_array.hpp>

#include <spray/core/camera.hpp>
#include <spray/core/world_base.hpp>

#include <spray/xyz/xyz.hpp>

#include <chrono>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        spray::log(spray::log_level::error, "usage: ./spray filename.xyz");
        return 1;
    }

    const auto glfw   = spray::glfw::init();
    auto window = spray::glfw::window<std::unique_ptr>(1024, 512, "spray");
    spray::glfw::make_context_current(window);

    // load OpenGL function as a function pointer at runtime.
    // this should be called after creating OpenGL context.
    spray::glad::load(glfwGetProcAddress);

    // set vsync interval. this should be called after making context
    spray::glfw::swap_interval(1);

    spray::core::buffer_array bufarray(spray::glfw::get_frame_buffer_size(window));

    cudaStream_t stream;
    spray::core::cuda_assert(
            cudaStreamCreateWithFlags(&stream, cudaStreamDefault));

    auto wld = spray::core::make_world();

    auto bufsize = spray::glfw::get_frame_buffer_size(window);
    auto cam = spray::core::make_pinhole_camera(
            "default pinhole camera",
            /* loc = */spray::geom::make_point(0.0, 0.0,  0.0),
            /* dir = */spray::geom::make_point(0.0, 0.0, -1.0),
            /* vup = */spray::geom::make_point(0.0, 1.0,  0.0),
            90.0f, bufsize.first, bufsize.second);

    auto xyz_reader = spray::xyz::reader(argv[1]);
    const auto snapshot = xyz_reader.read_snapshot(0);
    for(const auto p : snapshot.particles)
    {
        if(p.name == "C")
        {
            wld->push_back(spray::geom::make_sphere(
                spray::geom::make_point(p.vec[0], p.vec[1], p.vec[2]), 1.7f),
                spray::core::material{spray::core::make_color(0.5, 0.5, 0.5)});
        }
        else if(p.name == "H")
        {
            wld->push_back(spray::geom::make_sphere(
                spray::geom::make_point(p.vec[0], p.vec[1], p.vec[2]), 1.1f),
                spray::core::material{spray::core::make_color(1.0, 1.0, 1.0)});
        }
    }

    window.set_camera(cam.get());
    window.set_world (wld.get());
    window.set_bufarray(std::addressof(bufarray));

    spray::glfw::set_key_callback(window,
        [](GLFWwindow* win, int key, int code, int action, int mods) -> void {
            const auto wp = reinterpret_cast<spray::glfw::window_parameter*>(
                    glfwGetWindowUserPointer(win));
            if(!wp->is_focused) {return;}
            if(!wp->camera)     {return;}

            switch(key)
            {
                case GLFW_KEY_W    : wp->camera->advance( 0.1f);  break;
                case GLFW_KEY_S    : wp->camera->advance(-0.1f);  break;
                case GLFW_KEY_A    : wp->camera->lateral(-0.1f);  break;
                case GLFW_KEY_D    : wp->camera->lateral( 0.1f);  break;
                case GLFW_KEY_UP   : wp->camera->pitch  ( 0.01f); break;
                case GLFW_KEY_DOWN : wp->camera->pitch  (-0.01f); break;
                case GLFW_KEY_LEFT : wp->camera->yaw    ( 0.01f); break;
                case GLFW_KEY_RIGHT: wp->camera->yaw    (-0.01f); break;
            }
            return;
        });

    spray::glfw::set_frame_buffer_size_callback(window,
        [](GLFWwindow* win, int w, int h) -> void {
            const auto wp = reinterpret_cast<spray::glfw::window_parameter*>(
                    glfwGetWindowUserPointer(win));
            if(!wp->bufarray) {return;}
            if(!wp->camera) {return;}

            wp->bufarray->resize(w, h);
            wp->camera->resize(w, h);
            return;
        });


    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window.get(), true);
    ImGui_ImplOpenGL3_Init(/*GLSL version*/"#version 130");

    while(!spray::glfw::should_close(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        bufsize = spray::glfw::get_frame_buffer_size(window);

        const dim3 threads(32, 32);
        const dim3 blocks (static_cast<int>(std::ceil(double(bufsize.first)  / threads.x)),
                           static_cast<int>(std::ceil(double(bufsize.second) / threads.y)));

        fmt::print("\r32x32 threads for {}x{} blocks = {}x{} threads for {}x{} window "
                   "(bufsize / threads.x = {}, bufsize / threads.y = {})",
                   blocks.x, blocks.y, blocks.x * threads.x, blocks.y * threads.y,
                   bufsize.first, bufsize.second,
                   double(bufsize.first)  / threads.x,
                   double(bufsize.second) / threads.y
                   );

        cam->render(blocks, threads, stream, *wld, bufarray);
        spray::core::blit_framebuffer(bufarray);

        const bool cam_subwin_focused = cam->update_gui();
        window.set_is_focused(cam_subwin_focused);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        spray::glfw::swap_buffers(window);

    }
    return 0;
}
