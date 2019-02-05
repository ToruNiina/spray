#ifndef SPRAY_UTIL_SMART_PTR_DISPATCHER_HPP
#define SPRAY_UTIL_SMART_PTR_DISPATCHER_HPP
#include <utility>
#include <memory>

namespace spray
{
namespace util
{

// XXX: note about the weird implementation
// Firstly, std::shared_ptr takes only one template parameter, T. It means that
// it does not recieve Deleter type. It uses type erasure technique to store
// arbitrary deleter inside. In order to use both `std::unique/shared_ptr`,
// we need to make a wrapper to define SmartPtr<T, D>.
//
// std::shared_ptr::reset and the constructor require the contained type is a
// complete type. However, SDL_Window never be a complete type (because it is
// defined in a library and the actual implementation might be changed by
// updating the library). Therefore, we cannot use SmartPtr::reset() because
// SmartPtr may be `std::shared_ptr`. To deal with this problem, first we
// construct unique_ptr and then move it to a general SmartPtr. Note that we
// can assign unique_ptr<T, D> into shared_ptr<T>.

template<template<typename ...> class SmartPtr,
         typename Resource, typename Deleter>
struct smart_ptr_dispatcher;

template<typename Resource, typename Deleter>
struct smart_ptr_dispatcher<std::unique_ptr, Resource, Deleter>
{
    using type = std::unique_ptr<Resource, Deleter>;

    static type make(Resource* res, Deleter&& del) noexcept
    {
        return std::unique_ptr<Resource, Deleter>(res, std::move(del));
    }
    static type make(Resource* res, const Deleter& del) noexcept
    {
        return std::unique_ptr<Resource, Deleter>(res, del);
    }
};

template<typename Resource, typename Deleter>
struct smart_ptr_dispatcher<std::shared_ptr, Resource, Deleter>
{
    using type = std::shared_ptr<Resource>;

    static type make(Resource* res, Deleter&& del) noexcept
    {
        return type(std::unique_ptr<Resource, Deleter>(res, std::move(del)));
    }
    static type make(Resource* res, const Deleter& del) noexcept
    {
        return type(std::unique_ptr<Resource, Deleter>(res, del));
    }
};

template<template<typename ...> class SmartPtr, typename T, typename D>
using smart_ptr_t = typename smart_ptr_dispatcher<SmartPtr, T, D>::type;

template<template<typename ...> class SmartPtr, typename T, typename D>
smart_ptr_t<SmartPtr, T, D> make_ptr(T* ptr, D&& deleter)
{
    return smart_ptr_dispatcher<SmartPtr, T, D>::make(ptr, std::forward<D>(deleter));
}

} // util
} // spray
#endif // SPRAY_UTIL_SMART_PTR_DISPATCHER_HPP
