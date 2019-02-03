#ifndef SPRAY_SDL_LOCKED_TEXTURE_VIEW_HPP
#define SPRAY_SDL_LOCKED_TEXTURE_VIEW_HPP
#include <SDL.h>
#include <type_traits>

namespace spray
{
namespace sdl
{

// forward decl for pointer
template<template<typename ...> class SmartPtr>
class texture;

template<typename Pixel>
struct locked_texture_view
{
  public:

    using backend_texture_type = SDL_Texture;
    using pixel_type = Pixel;
    static_assert(std::is_trivially_copyable<pixel_type>::value, "");

    locked_texture_view(backend_texture_type* tex,
                        pixel_type* pixels, int pitch) noexcept
        : texture_(tex), pixels_(pixels), pitch_(pitch)
    {}
    ~locked_texture_view() noexcept
    {
        SDL_UnlockTexture(this->texture_);
    }

    void write(std::size_t x, std::size_t y, pixel_type px) const noexcept
    {
        auto dst = reinterpret_cast<unsigned char*>(this->pixels_);
        dst += y * pitch_ + x * sizeof(pixel_type);
        std::memcpy(dst, std::addressof(px), sizeof(pixel_type));
        return;
    }

    // std::launder...
    pixel_type* operator[](std::size_t line) const noexcept
    {
        auto bytes = reinterpret_cast<unsigned char*>(this->pixels_);
        return reinterpret_cast<pixel_type*>(bytes + line * pitch_);
    }

    bool    is_ok() const noexcept {return static_cast<bool>(this->pixels_);}
    operator bool() const noexcept {return this->is_ok();}

  private:

    backend_texture_type* texture_;
    pixel_type*           pixels_;
    int                   pitch_;
};

} // sdl
} // spray
#endif //  SPRAY_SDL_LOCKED_TEXTURE_VIEW_HPP
