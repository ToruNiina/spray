#ifndef SPRAY_UTIL_SMART_PTR_DISPATCHER_HPP
#define SPRAY_UTIL_SMART_PTR_DISPATCHER_HPP

#include <memory>

namespace spray
{
namespace util
{

template<template<typename ...> class SmartPtr,
         typename Resource, typename Deleter>
struct smart_ptr_dispatcher;

template<typename Resource, typename Deleter>
struct smart_ptr_dispatcher<std::unique_ptr, Resource, Deleter>
{
    using type = std::unique_ptr<Resource, Deleter>;
};

template<typename Resource, typename Deleter>
struct smart_ptr_dispatcher<std::shared_ptr, Resource, Deleter>
{
    using type = std::shared_ptr<Resource>;
};

template<template<typename ...> class SmartPtr, typename T, typename D>
using smart_ptr_t = typename smart_ptr_dispatcher<SmartPtr, T, D>::type;

} // util
} // spray
#endif // SPRAY_UTIL_SMART_PTR_DISPATCHER_HPP
