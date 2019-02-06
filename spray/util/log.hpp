#ifndef SPRAY_UTIL_LOG_HPP
#define SPRAY_UTIL_LOG_HPP
#include <fmt/format.h>
#include <fmt/color.h>
#include <map>
#include <cstdint>

namespace spray
{

enum class log_level : std::uint8_t
{
    error,
    warn ,
    info ,
    debug,
};

namespace detail
{

template<typename Level>
struct logger
{
    static bool is_activated(const Level lv)
    {
        const auto found = filter.find(lv);
        if(found == filter.end()){return true;}
        return found->second;
    }

    static void activate  (const Level lv) {filter[lv] = true;}
    static void inactivate(const Level lv) {filter[lv] = false;}

  private:
    static std::map<Level, bool> filter;
};
template<typename Level>
std::map<Level, bool> logger<Level>::filter;

} // detail

template<typename S, typename ... Args>
void log(log_level level, const S& format, const Args& ... args)
{
    if(!detail::logger<log_level>::is_activated(level)) {return;}

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
