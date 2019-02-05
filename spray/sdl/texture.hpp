#ifndef SPRAY_SDL_TEXTURE_HPP
#define SPRAY_SDL_TEXTURE_HPP
#include <spray/sdl/window.hpp>
#include <spray/sdl/rectangle.hpp>
#include <spray/sdl/locked_texture_view.hpp>
#include <spray/util/log.hpp>
#include <SDL.h>
#include <memory>

namespace spray
{
namespace sdl
{

template<template<typename ...> class SmartPtr>
class texture
{
  public:

    using backend_type  = SDL_Texture;
    using deleter_type  = decltype(&SDL_DestroyTexture);
    using resource_type = spray::util::smart_ptr_t<
        SmartPtr, backend_type, deleter_type>;

    template<template<typename ...> class Ptr>
    texture(const window<Ptr>& win,
            std::uint32_t format, int access, int w, int h)
        : texture_(spray::util::make_ptr<SmartPtr, SDL_Texture>(
                   nullptr, &SDL_DestroyTexture))
    {
        auto tex = SDL_CreateTexture(win.backend_renderer_ptr(),
                                     format, access, w, h);
        if(tex == nullptr)
        {
            spray::log(spray::log_level::error, "SDL_CreateTexture "
                       "for Window \"{}\" failed.\n", win.title());
            spray::log(spray::log_level::error, "=> {}\n", SDL_GetError());

            throw std::runtime_error(fmt::format(
                "Error: SDL_CreateTexture failed.\nError: => {}", SDL_GetError()));
        }

        this->texture_ = spray::util::make_ptr<SmartPtr>(tex, &SDL_DestroyTexture);
    }

    template<typename Pixel>
    locked_texture_view<Pixel> lock(const rectangle& rect) const noexcept
    {
        if(this->access() != SDL_TEXTUREACCESS_STREAMING) // SDL bug #1586
        {
            spray::log(spray::log_level::error,
                       "texture is not marked as STREAMING.\n");
            return locked_texture_view<Pixel>(nullptr, nullptr, 0);
        }

        Pixel* pixels = nullptr;
        int pitch = 0;
        const auto result = SDL_LockTexture(this->texture_.get(),
                std::addressof(rect),
                reinterpret_cast<void**>(std::addressof(pixels)),
                std::addressof(pitch));
        if(result != 0)
        {
            spray::log(spray::log_level::error, "SDL_LockTexture failed.\n");
            spray::log(spray::log_level::error, "=> {}\n", SDL_GetError());
            return locked_texture_view<Pixel>(nullptr, nullptr, 0);
        }
        return locked_texture_view<Pixel>(texture_.get(), pixels, pitch);
    }

    std::pair<int, int> size() const noexcept
    {
        const auto current = this->query();
        return std::make_pair(std::get<2>(current), std::get<3>(current));
    }
    int access() const noexcept
    {
        return std::get<1>(this->query());
    }
    std::uint32_t format() const noexcept
    {
        return std::get<0>(this->query());
    }

    std::tuple<std::uint32_t, int, int, int> query() const noexcept
    {
        std::uint32_t format = 0;
        int access = 0, w = 0, h = 0;
        SDL_QueryTexture(this->texture_.get(), &format, &access, &w, &h);
        return std::make_tuple(format, access, w, h);
    }

    backend_type* backend_texture_ptr() const noexcept {return texture_.get();}

  private:

    resource_type texture_;
};

template<template<typename ...> class Ptr1, template<typename ...> class Ptr2>
bool render(const window<Ptr1>& win, const texture<Ptr2>& tex) noexcept
{
    const auto result = SDL_RenderCopy(win.backend_renderer_ptr(),
            tex.backend_texture_ptr(), nullptr, nullptr);
    if(result != 0)
    {
        spray::log(spray::log_level::error, "failed to render a texture.\n");
        spray::log(spray::log_level::error, "=> {}\n", SDL_GetError());
        return false;
    }
    return true;
}

} // sdl
} // spray
#endif // SPRAY_SDL_TEXTURE_HPP
