add_library(imgui STATIC ${IMGUI_SRC})
add_library(glad  STATIC ${GLAD_SRC})

cuda_add_executable(spray main.cu ${IMGUI_SRC} ${GLAD_SRC})
set_target_properties(spray
    PROPERTIES
    COMPILE_FLAGS "-std=c++14 -O2 -Wall -Wpedantic -Wextra"
)
target_link_libraries(spray
    imgui
    glad
    ${GLFW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    dl
)
