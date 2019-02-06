#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <spray/util/unique_resource.hpp>

namespace test
{
static bool fptr_called = false;
void fptr() {fptr_called = true;}
}

TEST_CASE("unique_resoruce for lambda function", "[unique_resource_lambda]")
{
    bool called = false;
    {
        auto unique_resource = spray::util::make_unique_resource(
                std::ref(called), [](bool& b){b = true;});
    }
    REQUIRE(called);
}

