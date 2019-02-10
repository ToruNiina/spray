#ifndef SPRAY_UTIL_OBSERVER_PTR
#define SPRAY_UTIL_OBSERVER_PTR
#include <type_traits>

namespace spray
{
namespace util
{

template<typename T>
class observer_ptr
{
  public:
    typedef T value_type;
    typedef value_type* pointer;
    typedef value_type& reference;
    typedef value_type const* const_pointer;
    typedef value_type const& const_reference;

    constexpr observer_ptr() noexcept : ptr(nullptr) {}
    ~observer_ptr() = default;
    observer_ptr(observer_ptr const& rhs) = default;
    observer_ptr(observer_ptr&& rhs)      = default;
    observer_ptr& operator=(observer_ptr const& rhs) = default;
    observer_ptr& operator=(observer_ptr&& rhs)      = default;

    explicit observer_ptr(pointer p) : ptr(p){}
    observer_ptr& operator=(pointer p){ptr = p; return *this;}

    template<typename U, class = typename std::enable_if<
        std::is_convertible<U*, T*>::value>::type>
    explicit observer_ptr(U* p) : ptr(static_cast<T*>(p)){}
    template<typename U, class = typename std::enable_if<
        std::is_convertible<U*, T*>::value>::type>
    observer_ptr& operator=(U* p){ptr = static_cast<T*>(p); return *this;}

    constexpr pointer release() noexcept;
    constexpr void    reset(pointer p = nullptr) noexcept {this->ptr = p;}
    constexpr pointer get() const noexcept {return this->ptr;}
    constexpr void    swap(observer_ptr& other) noexcept;
    constexpr operator bool() const noexcept {return this->ptr != nullptr;}
    constexpr reference operator*() const {return *ptr;}
    constexpr pointer  operator->() const noexcept {return ptr;}
    constexpr explicit operator pointer() const noexcept {return this->ptr;}

  private:
    pointer ptr;
};

template<typename T>
inline constexpr typename observer_ptr<T>::pointer
observer_ptr<T>::release() noexcept
{
    const auto tmp = ptr;
    ptr = nullptr;
    return tmp;
}


template<typename T>
inline constexpr void observer_ptr<T>::swap(observer_ptr<T>& other) noexcept
{
    const auto tmp = this->ptr;
    this->ptr = other.get();
    other.reset(tmp);
    return;
}

template<typename T>
inline observer_ptr<T> make_observer(T* p) noexcept
{
    return observer_ptr<T>(p);
}

template<typename T, typename U>
inline bool operator==(const observer_ptr<T>& p1, const observer_ptr<U>& p2) noexcept
{
    return p1.get() == p2.get();
}

template<typename T, typename U>
inline bool operator!=(const observer_ptr<T>& p1, const observer_ptr<U>& p2) noexcept
{
    return !(p1 == p2);
}

template<typename T>
inline bool operator==(const observer_ptr<T>& p, std::nullptr_t) noexcept
{
    return !p;
}

template<typename T>
inline bool operator==(std::nullptr_t, const observer_ptr<T>& p) noexcept
{
    return !p;
}

template<typename T>
inline bool operator!=(const observer_ptr<T>& p, std::nullptr_t) noexcept
{
    return static_cast<bool>(p);
}

template<typename T>
inline bool operator!=(std::nullptr_t, const observer_ptr<T>& p) noexcept
{
    return static_cast<bool>(p);
}

template<typename T, typename U>
bool operator<(const observer_ptr<T>& p1, const observer_ptr<U>& p2) noexcept
{
    using common_type = typename std::common_type<
        typename observer_ptr<T>::pointer,
        typename observer_ptr<U>::pointer>::type;
    return common_type(p1.get()) < common_type(p2.get());
}

template<typename T, typename U>
bool operator>(const observer_ptr<T>& p1, const observer_ptr<U>& p2) noexcept
{
    return p2 < p1;
}

template<typename T, typename U>
bool operator<=(const observer_ptr<T>& p1, const observer_ptr<U>& p2) noexcept
{
    return !(p2 < p1);
}

template<typename T, typename U>
bool operator>=(const observer_ptr<T>& p1, const observer_ptr<U>& p2) noexcept
{
    return !(p1 < p2);
}

template<typename T>
inline void swap(const observer_ptr<T>& lhs, const observer_ptr<T>& rhs) noexcept
{
    return lhs.swap(rhs);
}

}// util
}// spray
#endif// SPRAY_UTIL_OBSERVER_PTR
