#include <spray/glad/load.hpp>
#include <spray/glfw/init.hpp>
#include <spray/glfw/window.hpp>

#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_opengl3.h>

int main()
{
    const auto glfw   = spray::glfw::init();
    const auto window = spray::glfw::window<std::unique_ptr>(640, 480, "spray");

    spray::glfw::make_context_current(window);
    spray::glfw::swap_interval(1);

    spray::glad::load(glfwGetProcAddress);

    while(!spray::glfw::should_close(window))
    {
        glfwPollEvents();
    }
    return 0;
}
