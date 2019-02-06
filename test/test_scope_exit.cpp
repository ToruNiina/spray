#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <spray/util/scope_exit.hpp>

TEST_CASE("scope_exit", "[scope_exit]")
{
    bool called = false;
    {
        auto make_it_true = [&](){called = true;};
        const auto scope_exit = spray::util::make_scope_exit(std::move(make_it_true));
    }
    REQUIRE(called);
}
