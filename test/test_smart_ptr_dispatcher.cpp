#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <spray/util/smart_ptr_dispatcher.hpp>

namespace test
{
static bool called = false;
void f(int* p)
{
    called = true;
    delete p;
}
} // test

TEST_CASE("smart_ptr_t for shared_ptr", "[smart_ptr_t<shared_ptr>]")
{
    static_assert(std::is_same<
            spray::util::smart_ptr_t<std::shared_ptr, int, decltype(&test::f)>,
            std::shared_ptr<int>>::value, "");

    test::called = false;
    REQUIRE(!test::called);
    {
        const auto ptr = spray::util::make_ptr<std::shared_ptr>(new int(42), &test::f);
    }
    REQUIRE(test::called);

}

TEST_CASE("scope_exit for function pointer", "[scope_exit_fptr]")
{
    static_assert(std::is_same<
            spray::util::smart_ptr_t<std::unique_ptr, int, decltype(&test::f)>,
            std::unique_ptr<int, decltype(&test::f)>>::value, "");
    test::called = false;
    REQUIRE(!test::called);
    {
        const auto ptr = spray::util::make_ptr<std::shared_ptr>(new int(42), &test::f);
    }
    REQUIRE(test::called);
}
