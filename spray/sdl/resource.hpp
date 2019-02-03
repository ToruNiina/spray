#ifndef SPRAY_SDL_RESOURCE_HPP
#define SPRAY_SDL_RESOURCE_HPP
#include <spray/util/log.hpp>
#include <SDL.h>

namespace spray
{
namespace sdl
{

struct resource
{
    resource()
    {
        const auto result = SDL_Init(SDL_INIT_EVERYTHING);
        if(result != 0)
        {
            spray::log(spray::log_level::error, "SDL_Init failed.\n");
            spray::log(spray::log_level::error, "=> {}\n", SDL_GetError());

            throw std::runtime_error(fmt::format(
                "Error: SDL_Init failed.\nError: => {}", SDL_GetError()));
        }
        spray::log(spray::log_level::info, "SDL initialized.\n");
    }
    ~resource()
    {
        SDL_Quit();
    }
};

} // sdl
} // spray
#endif// SPRAY_SDL_RESOURCE_HPP
