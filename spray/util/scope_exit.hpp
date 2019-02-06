#ifndef SPRAY_UTIL_SCOPE_EXIT_HPP
#define SPRAY_UTIL_SCOPE_EXIT_HPP
#include <type_traits>
#include <utility>

namespace spray
{
namespace util
{

template<typename EF>
struct scope_exit
{
  public:
    using exit_function_type = EF;

    template<typename EFP>
    explicit scope_exit(EFP&& func)
        noexcept(std::is_nothrow_constructible<EF, EFP>::value ||
                 std::is_nothrow_constructible<EF, EFP&>::value)
        : func_(std::forward<EFP>(func)), execute_on_destruction_(true)
    {}
    ~scope_exit() noexcept
    {
        if(this->execute_on_destruction_) {this->func_();}
    }

    scope_exit(scope_exit&& other)
        noexcept(std::is_nothrow_move_constructible<exit_function_type>::value)
        : func_(std::move(other.func_)),
          execute_on_destruction_(other.execute_on_destruction_)
    {}
    scope_exit(const scope_exit&)            = delete;
    scope_exit& operator=(const scope_exit&) = delete;
    scope_exit& operator=(scope_exit&&)      = delete;

    void release() noexcept {this->execute_on_destruction_ = false;}

  private:

     exit_function_type func_;
     bool execute_on_destruction_;
};

template<typename EF>
auto make_scope_exit(EF&& exit_func)
    noexcept((std::is_nothrow_move_constructible<EF>::value &&
              std::is_rvalue_reference<decltype(exit_func)>::value) ||
             (std::is_nothrow_copy_constructible<EF>::value &&
              std::is_lvalue_reference<decltype(exit_func)>::value))
{
    return scope_exit<std::remove_reference_t<EF>>(std::forward<EF>(exit_func));
}

} // util
} // spray
#endif// SPRAY_UTIL_SCOPE_EXIT_HPP
