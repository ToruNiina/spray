add_library(imgui STATIC ${IMGUI_SRC})
add_library(glad  STATIC ${GLAD_SRC})

set(SPRAY_CPP_SRC
    ${PROJECT_SOURCE_DIR}/spray/core/save_image.cpp
)

set(SPRAY_CUDA_SRC
    ${PROJECT_SOURCE_DIR}/spray/core/world.cu
    ${PROJECT_SOURCE_DIR}/spray/core/pinhole_camera.cu
#     ${PROJECT_SOURCE_DIR}/spray/core/orthogonal_camera.cu
    ${PROJECT_SOURCE_DIR}/spray/core/show_image.cu
)
set(CUDA_NVCC_FLAGS "-std=c++14;-O2;--expt-extended-lambda")

cuda_add_executable(spray main.cpp ${SPRAY_CPP_SRC} ${SPRAY_CUDA_SRC})
set_target_properties(spray PROPERTIES
    COMPILE_FLAGS "-Wall -Wpedantic -Wextra"
)
target_link_libraries(spray
    imgui
    glad
    ${GLFW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${LIBPNG_LIBRARIES}
    dl
)
