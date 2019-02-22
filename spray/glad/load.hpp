#ifndef SPRAY_GLAD_LOAD_HPP
#define SPRAY_GLAD_LOAD_HPP
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <spray/util/log.hpp>

namespace spray
{
namespace glad
{

inline void load()
{
    const auto result = gladLoadGL();
    if(result != 0)
    {
        spray::log(spray::log_level::error, "failed to load OpenGL via glad");
        throw std::runtime_error("Error: failed to load OpenGL via glad");
    }
    spray::log(spray::log_level::info, "glad loaded OpenGL ",
               GLVersion.major, '.', GLVersion.minor, '\n');
    return;
}

inline void load(decltype(&glfwGetProcAddress) fptr)
{
    const auto version = gladLoadGLLoader(reinterpret_cast<GLADloadproc>(fptr));
    if(version == 0)
    {
        spray::log(spray::log_level::error, "failed to load OpenGL via glad");
        throw std::runtime_error("Error: failed to load OpenGL via glad");
    }
    spray::log(spray::log_level::info, "glad loaded OpenGL ",
               GLVersion.major, '.', GLVersion.minor, '\n');
    return;
}


} // glad
} // spray
#endif// SPRAY_GLAD_LOAD_HPP
