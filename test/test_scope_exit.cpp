#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <spray/util/scope_exit.hpp>

namespace test
{
static bool fptr_called = false;
void fptr() {fptr_called = true;}
}

TEST_CASE("scope_exit for lambda function", "[scope_exit_lambda]")
{
    bool called = false;
    {
        auto scope_exit = spray::util::make_scope_exit([&](){called = true;});
    }
    REQUIRE(called);
}

TEST_CASE("scope_exit for function pointer", "[scope_exit_fptr]")
{
    test::fptr_called = false;
    {
        const auto scope_exit = spray::util::make_scope_exit(&test::fptr);
    }
    REQUIRE(test::fptr_called);
}
