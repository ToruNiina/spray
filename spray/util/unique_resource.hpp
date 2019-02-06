#ifndef SPRAY_UTIL_UNIQUE_RESOURCE_HPP
#define SPRAY_UTIL_UNIQUE_RESOURCE_HPP
#include <type_traits>
#include <utility>

namespace spray
{
namespace util
{

template<typename R, typename D>
struct unique_resource
{
  public:
    using resource_type = R;
    using deleter_type  = D;

  public:

    unique_resource(resource_type&& resource, deleter_type&& deleter,
                    bool deletion_needed = true)
        noexcept(std::is_nothrow_move_constructible<resource_type>::value &&
                 std::is_nothrow_move_constructible<deleter_type>::value)
        : resource_(std::move(resource)), deleter_ (std::move(deleter)),
          deletion_needed_(deletion_needed)
    {}
    ~unique_resource() noexcept(
        noexcept(std::declval<deleter_type>()(std::declval<resource_type>())))
    {
        this->reset();
    }

    unique_resource(unique_resource&& other)
        noexcept(std::is_nothrow_move_constructible<resource_type>::value &&
                 std::is_nothrow_move_constructible<deleter_type>::value)
        : resource_(std::move(other.resource_)),
          deleter_(std::move(other.deleter_)),
          deletion_needed_(other.deletion_needed_)
    {
        other.release();
    }

    unique_resource& operator=(unique_resource&& other)
        noexcept(std::is_nothrow_move_assignable<resource_type>::value &&
                 std::is_nothrow_move_assignable<deleter_type>::value)
    {
        this->reset();
        this->resource_ = std::move(other.resource_);
        this->deleter_  = std::move(other.deleter_);
        this->deletion_needed_ = other.deletion_needed_;
        other.release();
        return *this;
    }

    unique_resource() = delete;
    unique_resource(const unique_resource&) = delete;
    unique_resource& operator=(const unique_resource&) = delete;

    void reset() noexcept(
        noexcept(std::declval<deleter_type>()(std::declval<resource_type>())))
    {
        if(deletion_needed_) {this->deleter_(this->resource_);}
        this->deletion_needed_ = false;
    }

    void reset(resource_type&& new_resource) noexcept(
        noexcept(std::declval<deleter_type>()(std::declval<resource_type>())) &&
        std::is_nothrow_move_assignable<resource_type>::value)
    {
        if(deletion_needed_) {this->deleter_(this->resource_);}
        this->resource_ = std::move(new_resource);
        this->deletion_needed_ = true;
    }

    resource_type const& release() noexcept
    {
        this->deletion_needed_ = false;
        return this->resource_;
    }

    operator resource_type const&() const noexcept {return this->resource_;}

    resource_type const& get()         const noexcept {return this->resource_;}
    deleter_type  const& get_deleter() const noexcept {return this->deleter_;}

  private:

    resource_type resource_;
    deleter_type  deleter_;
    bool deletion_needed_;
};

template<typename R, typename D>
auto make_unique_resource(R&& resource, D&& deleter)
{
    static_assert(std::is_rvalue_reference<decltype(resource)>::value,
        "unique_resource only accepts an r-value reference of a resource");
    static_assert(std::is_rvalue_reference<decltype(deleter)>::value,
        "unique_resource only accepts an r-value reference of a deleter");
    using result_type = unique_resource<
        std::remove_reference_t<R>, std::remove_reference_t<D>>;
    return result_type(std::move(resource), std::move(deleter));
}

} // util
} // spray
#endif // SPRAY_UTIL_UNIQUE_RESOURCE_HPP
