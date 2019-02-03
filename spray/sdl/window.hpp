#ifndef SPRAY_SDL_WINDOW_HPP
#define SPRAY_SDL_WINDOW_HPP
#include <spray/util/smart_ptr_dispatcher.hpp>
#include <spray/util/log.hpp>
#include <SDL.h>
#include <memory>

namespace spray
{
namespace sdl
{

template<template<typename ...> class SmartPtr>
class window
{
  public:

    using window_backend_type  = SDL_Window;
    using window_deleter_type  = decltype(&SDL_DestroyWindow);
    using window_resource_type = spray::util::smart_ptr_t<
        SmartPtr, window_backend_type, window_deleter_type>;

    using renderer_backend_type  = SDL_Renderer;
    using renderer_deleter_type  = decltype(&SDL_DestroyRenderer);
    using renderer_resource_type = spray::util::smart_ptr_t<
        SmartPtr, renderer_backend_type, renderer_deleter_type>;

    // XXX: note about the weird implementation
    // std::shared_ptr::reset and the constructor require the contained type
    // is a complete type. However, SDL_Window never be a complete type
    // (because it is defined in a library and the actual implementation
    //  might be changed by updating the library). Therefore, we cannot use
    // SmartPtr::reset() because SmartPtr may be `std::shared_ptr`. To
    // deal with this problem, first we construct unique_ptr and then move
    // it to a general SmartPtr. Note that we can assign unique_ptr<T> into
    // shared_ptr<T>.

    using unique_window_ptr =
        std::unique_ptr<window_backend_type,   window_deleter_type  >;
    using unique_renderer_ptr =
        std::unique_ptr<renderer_backend_type, renderer_deleter_type>;

  public:

    window(std::string title, std::size_t width, std::size_t height)
        : title_(std::move(title)),
          window_  (unique_window_ptr  (nullptr, &SDL_DestroyWindow)),
          renderer_(unique_renderer_ptr(nullptr, &SDL_DestroyRenderer))
    {
        auto win = SDL_CreateWindow(title.c_str(),
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height,
            SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
        if(win == nullptr)
        {
            spray::log(spray::log_level::error, "SDL_CreateWindow "
                "for \"{}\" failed.\n", this->title_);
            spray::log(spray::log_level::error, "=> {}\n", SDL_GetError());

            throw std::runtime_error(fmt::format(
                "Error: SDL_CreateWindow for \"{}\" failed.\nError: => {}",
                this->title_, SDL_GetError()));
        }
        spray::log(spray::log_level::info, "Window \"{}\" created.\n", title_);

        auto ren = SDL_CreateRenderer(win, -1, 0);
        if(ren == nullptr)
        {
            spray::log(spray::log_level::error, "SDL_CreateRenderer "
                "for \"{}\" failed.\n", this->title_);
            spray::log(spray::log_level::error, "=> {}\n", SDL_GetError());

            throw std::runtime_error(fmt::format(
                "Error: SDL_CreateWindow for \"{}\" failed.\nError: => {}",
                this->title_, SDL_GetError()));
        }
        spray::log(spray::log_level::info,
                   "Renderer for Window \"{}\" created.\n", this->title_);


        this->window_   = unique_window_ptr  (win, &SDL_DestroyWindow);
        this->renderer_ = unique_renderer_ptr(ren, &SDL_DestroyRenderer);
    }

    void update()
    {
        SDL_RenderPresent(renderer_.get());
    }

    std::string const& title() const noexcept {return title_;}

    window_backend_type*   backend_window_ptr()   const noexcept
    {return window_.get();}
    renderer_backend_type* backend_renderer_ptr() const noexcept
    {return renderer_.get();}

  private:

    std::string            title_;
    window_resource_type   window_;
    renderer_resource_type renderer_;
};

} // sdl
} // spray
#endif // SPRAY_SDL_WINDOW_HPP
