add_executable(spray main.cpp ${IMGUI_SRC} ${GL3W_SRC})
set_target_properties(spray
    PROPERTIES
    COMPILE_FLAGS "-std=c++14 -O2 -Wall -Wpedantic -Wextra"
)
target_link_libraries(spray
    ${SDL2_LIBRARIES}
    ${OPENGL_LIBRARIES}
    dl
)
