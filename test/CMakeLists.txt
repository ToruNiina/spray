set(TEST_NAMES
    test_scope_exit
    test_unique_resource
    test_smart_ptr_dispatcher
    )

foreach(TEST_NAME ${TEST_NAMES})
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    set_target_properties(${TEST_NAME} PROPERTIES
        COMPILE_FLAGS "-std=c++14 -O2")
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME} WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endforeach(TEST_NAME)
