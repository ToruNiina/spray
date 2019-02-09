#ifndef SPRAY_GLFW_WINDOW_HPP
#define SPRAY_GLFW_WINDOW_HPP
#include <spray/util/smart_ptr_dispatcher.hpp>
#include <spray/util/log.hpp>
#include <GLFW/glfw3.h>

namespace spray
{
namespace glfw
{

template<template<typename...> class SmartPtr>
struct window
{
  public:
    using window_type   = GLFWwindow;
    using deleter_type  = decltype(&glfwDestroyWindow);
    using resource_type =
        spray::util::smart_ptr_t<SmartPtr, window_type, deleter_type>;

    window(std::size_t w, std::size_t h, const char* name)
        : resource_(spray::util::make_ptr<SmartPtr, window_type>(
                        nullptr, &glfwDestroyWindow)),
          name_(name)
    {
        glfwWindowHint(GLFW_DEPTH_BITS,            0);
        glfwWindowHint(GLFW_STENCIL_BITS,          0);
        glfwWindowHint(GLFW_SRGB_CAPABLE,          GL_TRUE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);
        auto win = glfwCreateWindow(w, h, name, nullptr, nullptr);
        if(win == nullptr)
        {
            spray::log(spray::log_level::error, "failed to create glfw window");
            throw std::runtime_error("Error: failed to create glfw window");
        }
        this->resource_ = spray::util::make_ptr<SmartPtr>(win, &glfwDestroyWindow);
    }

    window_type* get() const noexcept {return resource_.get();}

  private:

    resource_type resource_;
    std::string name_;
};

template<template<typename...> class SmartPtr>
bool should_close(const window<SmartPtr>& win) noexcept
{
    return glfwWindowShouldClose(win.get());
}
template<template<typename...> class SmartPtr>
void make_context_current(const window<SmartPtr>& win) noexcept
{
    return glfwMakeContextCurrent(win.get());
}
template<template<typename...> class SmartPtr>
void swap_buffers(const window<SmartPtr>& win) noexcept
{
    glfwSwapBuffers(win.get());
    return;
}

template<template<typename...> class SmartPtr>
std::pair<int, int> get_frame_buffer_size(const window<SmartPtr>& win) noexcept
{
    int w, h;
    glfwGetFramebufferSize(win.get(), std::addressof(w), std::addressof(h));
    return std::make_pair(w, h);
}
template<template<typename...> class SmartPtr>
std::pair<int, int> size(const window<SmartPtr>& win) noexcept
{
    int w, h;
    glfwGetWindowSize(win.get(), std::addressof(w), std::addressof(h));
    return std::make_pair(w, h);
}

} // glfw
} // spray
#endif// SPRAY_GLFW_WINDOW_HPP
