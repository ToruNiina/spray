#ifndef SPRAY_UTIL_LOG_HPP
#define SPRAY_UTIL_LOG_HPP
#include <map>
#include <iostream>
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

inline void log_output(std::ostream&) noexcept
{
    return;
}
template<typename Arg, typename ... Args>
inline void log_output(std::ostream& os, const Arg& arg, const Args& ... args)
{
    os << arg;
    log_output(os, args...);
    return;
}
} // detail

template<typename ... Args>
void log(log_level level, const Args& ... args)
{
    if(!detail::logger<log_level>::is_activated(level)) {return;}
    switch(level)
    {
        case log_level::error:
        {
            std::cerr << "\x1b[31mError:\x1b[0m "; break;
        }
        case log_level::warn:
        {
            std::cerr << "\x1b[33mWarn:\x1b[0m "; break;
        }
        case log_level::info:
        {
            std::cerr << "\x1b[32mInfo:\x1b[0m "; break;
        }
        case log_level::debug:
        {
            std::cerr << "\x1b[34mDebug:\x1b[0m "; break;
        }
        default:
        {
            std::cerr << "\x1b[30;1mUnknown:\x1b[0m "; break;
        }
    }
    detail::log_output(std::cerr, args..., '\n');
    return;
}

} // spray
#endif // SPRAY_UTIL_LOG_HPP
