cmake_minimum_required(VERSION 3.0)
enable_testing()
project(spray)

set(CMAKE_CXX_STANDARD 14)

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extlib")
    execute_process(COMMAND mkdir extlib WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

# ----------------------------------------------------------------------------
# CUDA
find_package(CUDA REQUIRED)
if(${CUDA_FOUND})
    message(STATUS "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} found.")
else()
    message(FATAL_ERROR "CUDA could not be found.")
endif()
set(CUDA_SEPARABLE_COMPILATION ON)
cuda_include_directories(${PROJECT_SOURCE_DIR})

# ----------------------------------------------------------------------------
# OpenGL
find_package(OpenGL REQUIRED)
if(${OPENGL_FOUND})
    message(STATUS "OpenGL is found at ${OPENGL_INCLUDE_DIR}, ${OPENGL_LIBRARIES}")
else()
    message(FATAL_ERROR "OpenGL could not be found.")
endif()

# ----------------------------------------------------------------------------
# glfw3

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
if(${GLFW_FOUND})
    message(STATUS "glfw3 is found at ${GLFW_INCLUDE_DIRS}, ${GLFW_LIBRARIES}")
else()
    message(FATAL_ERROR "glfw3 could not be found.")
endif()
include_directories(${GLFW_INCLUDE_DIRS})

# ----------------------------------------------------------------------------
# glad

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extlib/glad")
    execute_process(COMMAND git clone https://github.com/Dav1dde/glad.git
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib")

    execute_process(COMMAND python -m glad --profile "core" --api "gl=4.5" --generator "c" --spec "gl" --extensions "" --out-path "."
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib/glad")
endif()
include_directories("${PROJECT_SOURCE_DIR}/extlib/glad/include")
set(GLAD_SRC "${PROJECT_SOURCE_DIR}/extlib/glad/src/glad.c")

# ----------------------------------------------------------------------------
# imgui

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extlib/imgui")
    execute_process(COMMAND git clone https://github.com/ocornut/imgui.git
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib")
    execute_process(COMMAND git checkout -b v1.67 refs/tags/v1.67
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib/imgui")
endif()

set(IMGUI_SRC
    "${PROJECT_SOURCE_DIR}/extlib/imgui/imgui.cpp"
    "${PROJECT_SOURCE_DIR}/extlib/imgui/imgui_draw.cpp"
    "${PROJECT_SOURCE_DIR}/extlib/imgui/imgui_demo.cpp"
    "${PROJECT_SOURCE_DIR}/extlib/imgui/imgui_widgets.cpp"
    "${PROJECT_SOURCE_DIR}/extlib/imgui/examples/imgui_impl_opengl3.cpp"
    "${PROJECT_SOURCE_DIR}/extlib/imgui/examples/imgui_impl_glfw.cpp"
)
include_directories("${PROJECT_SOURCE_DIR}/extlib/imgui")
include_directories("${PROJECT_SOURCE_DIR}/extlib/imgui/examples")
add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)

# ----------------------------------------------------------------------------
# libpng
include (FindPkgConfig)
if (NOT PKG_CONFIG_FOUND)
    message (FATAL_ERROR "pkg-config not found")
endif ()
pkg_check_modules(LIBPNG libpng12 REQUIRED)
if (NOT LIBPNG_FOUND)
    message(FATAL_ERROR "libpng-v1.2 not found")
else ()
    include_directories(${LIBPNG_INCLUDE_DIRS})
    link_directories   (${LIBPNG_LIBRARY_DIRS})
endif ()

# ----------------------------------------------------------------------------
# Catch2

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extlib/Catch2")
    execute_process(COMMAND git clone https://github.com/catchorg/Catch2.git
                    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib")
    execute_process(COMMAND git checkout -b v2.6.0 refs/tags/v2.6.0
                    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib/Catch2")
endif()
include_directories("${PROJECT_SOURCE_DIR}/extlib/Catch2/single_include")

# ----------------------------------------------------------------------------
# spray
include_directories(${PROJECT_SOURCE_DIR})
add_subdirectory(src)
add_subdirectory(test)
