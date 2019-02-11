#ifndef SPRAY_GLFW_WINDOW_HPP
#define SPRAY_GLFW_WINDOW_HPP
#include <spray/util/smart_ptr_dispatcher.hpp>
#include <spray/util/observer_ptr.hpp>
#include <spray/util/log.hpp>
#include <GLFW/glfw3.h>

namespace spray
{
namespace cuda { struct buffer_array; } // forward declaration
namespace core { struct camera_base; struct world_base;} // ditto
namespace glfw
{

struct window_parameter
{
    spray::util::observer_ptr<spray::cuda::buffer_array> bufarray;
    spray::util::observer_ptr<spray::core::camera_base>  camera;
    spray::util::observer_ptr<spray::core::world_base>   world;
    bool is_dragged; // true while mouse left button is pressed
    bool is_focused; // true while no imgui windows are selected
};

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

        this->params_.camera = nullptr;
        this->params_.world  = nullptr;
        this->params_.is_dragged = false;
        this->params_.is_focused = false;
        glfwSetWindowUserPointer(win, std::addressof(this->params_));

        this->resource_ = spray::util::make_ptr<SmartPtr>(win, &glfwDestroyWindow);
    }

    window_type* get() const noexcept {return resource_.get();}

    void set_camera(spray::core::camera_base* cam)
    {
        this->params_.camera.reset(cam);
    }
    void set_world(spray::core::world_base* wld)
    {
        this->params_.world.reset(wld);
    }
    void set_bufarray(spray::cuda::buffer_array* bufa)
    {
        this->params_.bufarray.reset(bufa);
    }

    void set_is_focused(bool flag) noexcept {this->params_.is_focused = flag;}
    void set_is_dragged(bool flag) noexcept {this->params_.is_dragged = flag;}

  private:

    resource_type    resource_;
    window_parameter params_;
    std::string      name_;
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

using key_callback          = void (*)(GLFWwindow*, int, int, int, int);
using mouse_button_callback = void (*)(GLFWwindow*, int, int, int);
using mouse_pos_callback    = void (*)(GLFWwindow*, double, double);
using scroll_callback       = void (*)(GLFWwindow*, double, double);

template<template<typename...> class SmartPtr>
void set_key_callback(
        const window<SmartPtr>& win, key_callback cb) noexcept
{
    glfwSetKeyCallback(win.get(), cb);
    return;
}
template<template<typename...> class SmartPtr>
void set_mouse_button_callback(
        const window<SmartPtr>& win, mouse_button_callback cb) noexcept
{
    glfwSetMouseButtonCallback(win.get(), cb);
    return;
}
template<template<typename...> class SmartPtr>
void set_mouse_pos_callback(
        const window<SmartPtr>& win, mouse_pos_callback cb) noexcept
{
    glfwSetCursorPosCallback(win.get(), cb);
    return;
}
template<template<typename...> class SmartPtr>
void set_scroll_callback(
        const window<SmartPtr>& win, scroll_callback cb) noexcept
{
    glfwSetScrollCallback(win.get(), cb);
    return;
}

} // glfw
} // spray
#endif// SPRAY_GLFW_WINDOW_HPP
