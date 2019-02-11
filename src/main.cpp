#include <spray/glad/load.hpp>
#include <spray/glfw/init.hpp>
#include <spray/glfw/window.hpp>
#include <spray/core/buffer_array.hpp>
#include <spray/core/render.hpp>

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
    auto window = spray::glfw::window<std::unique_ptr>(1200, 900, "spray");
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

    auto wld = spray::core::make_world();

    spray::core::pinhole_camera cam(
        /* loc = */spray::geom::make_point(0.0, 0.0,  0.0),
        /* dir = */spray::geom::make_point(0.0, 0.0, -1.0),
        /* vup = */spray::geom::make_point(0.0, 1.0,  0.0),
        90.0f,
        1200,
        900
        );

    auto xyz_reader = spray::xyz::reader(argv[1]);
    const auto snapshot   = xyz_reader.read_snapshot(0);

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

    window.set_camera(std::addressof(cam));
    window.set_world (wld.get());

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

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window.get(), true);
    ImGui_ImplOpenGL3_Init(/*GLSL version*/"#version 130");

    std::array<float, 3> cam_pos_buf{{0.0f, 0.0f, 0.0f}};
    std::array<float, 3> cam_dir_buf{{0.0f, 0.0f, 0.0f}};
    std::array<float, 3> cam_vup_buf{{0.0f, 0.0f, 0.0f}};

    while(!spray::glfw::should_close(window))
    {
        const auto start = std::chrono::system_clock::now();

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            ImGui::Begin("camera");
            window.set_is_focused(!ImGui::IsWindowFocused());

            const auto loc = cam.location();
            ImGui::Text("current camera position : %.3f %.3f %.3f",
                spray::geom::X(loc), spray::geom::Y(loc), spray::geom::Z(loc));

            ImGui::InputFloat3("camera position", cam_pos_buf.data());
            if(ImGui::Button("apply position"))
            {
                cam.move(spray::geom::make_point(
                         cam_pos_buf[0], cam_pos_buf[1], cam_pos_buf[2]));
            }

            const auto dir = cam.direction();
            ImGui::Text("current camera direction: %.3f %.3f %.3f",
                spray::geom::X(dir), spray::geom::Y(dir), spray::geom::Z(dir));

            ImGui::InputFloat3("camera direction", cam_dir_buf.data());
            if(ImGui::Button("apply direction"))
            {
                cam.look(spray::geom::unit(spray::geom::make_point(
                         cam_dir_buf[0], cam_dir_buf[1], cam_dir_buf[2])));
            }

            const auto framerate = ImGui::GetIO().Framerate;
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                        1000.0f / framerate, framerate);
            ImGui::End();
        }

        const auto size = spray::glfw::get_frame_buffer_size(window);
        const dim3 blocks (size.first / 32, size.second / 32);
        const dim3 threads(32, 32);
        spray::cuda::render(blocks, threads, stream, cam, *wld, bufarray);
        spray::cuda::blit_framebuffer(bufarray);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        spray::glfw::swap_buffers(window);

        const auto stop = std::chrono::system_clock::now();
        const auto uspf = std::chrono::duration_cast<std::chrono::microseconds>(
                stop - start).count();
        const auto fps  = 1.0e6 / uspf;

        fmt::print(fmt::fg(fmt::color::green), "\rInfo:");
        fmt::print(" {} fps", fps);
    }
    return 0;
}
