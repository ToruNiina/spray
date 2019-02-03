#ifndef SPRAY_UTIL_LOG_HPP
#define SPRAY_UTIL_LOG_HPP
#include <fmt/format.h>
#include <fmt/color.h>

namespace spray
{

enum class log_level
{
    error,
    warn,
    info,
    debug,
};

template<typename S, typename ... Args>
void log(log_level level, const S& format, const Args& ... args)
{
    switch(level)
    {
        case log_level::error:
        {
            ::fmt::print(::fmt::fg(::fmt::color::red),    "Error: "); break;
        }
        case log_level::warn:
        {
            ::fmt::print(::fmt::fg(::fmt::color::yellow), "Warn : "); break;
        }
        case log_level::info:
        {
            ::fmt::print(::fmt::fg(::fmt::color::green),  "Info : "); break;
        }
        case log_level::debug:
        {
            ::fmt::print(::fmt::fg(::fmt::color::blue),   "Debug: "); break;
        }
        default:
        {
            ::fmt::print(::fmt::fg(::fmt::color::orange), "Unknown: "); break;
        }
    }
    ::fmt::print(format, args...);
    return;
}

} // spray
#endif // SPRAY_UTIL_LOG_HPP
