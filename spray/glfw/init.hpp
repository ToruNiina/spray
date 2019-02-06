#ifndef SPRAY_GLFW_INIT_HPP
#define SPRAY_GLFW_INIT_HPP
#include <spray/util/scope_exit.hpp>
#include <spray/util/log.hpp>
#include <GLFW/glfw3.h>

namespace spray
{
namespace glfw
{

inline auto init()
{
    glfwSetErrorCallback([](int err, const char* desc) noexcept -> void {
        spray::log(spray::log_level::error, "({}): {}", err, desc);
        return;
    });

    const auto result = glfwInit();
    if(!result)
    {
        spray::log(spray::log_level::error, "failed to initialize glfw3");
        throw std::runtime_error("Error: failed to initialize glfw3");
    }
    return spray::util::make_scope_exit(&glfwTerminate);
}

} // glfw
} // spray
#endif// SPRAY_GLFW_INIT_HPP
