cmake_minimum_required(VERSION 3.0)
project(spray)

# ----------------------------------------------------------------------------
# OpenGL
find_package(OpenGL REQUIRED)
if(${OPENGL_FOUND})
    message(STATUS
        "OpenGL is found at ${OPENGL_INCLUDE_DIR}, ${OPENGL_LIBRARIES}")
endif()

# ----------------------------------------------------------------------------
# SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
if(${OPENGL_FOUND})
    message(STATUS "SDL2 is found at ${SDL2_INCLUDE_DIRS}, ${SDL2_LIBRARIES}")
endif()

# ----------------------------------------------------------------------------
# gl3w

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extlib/gl3w/include/GL/gl3w.h")
    execute_process(COMMAND "git clone https://github.com/skaslev/gl3w.git"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib")

    find_package (Python3 COMPONENTS Interpreter Development)
    if(Python3_FOUND)
        execute_process(COMMAND "${Python3_EXECUTABLE} gl3w_gen.py"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib/gl3w")
    else()
        find_package (Python2 COMPONENTS Interpreter Development)
        if(Python2_FOUND)
            if(${Python2_VERSION_MINOR} LESS 7)
                message(FATAL_ERROR "python 2.7/3.x not found.")
            endif()
            execute_process(COMMAND "${Python2_EXECUTABLE} gl3w_gen.py"
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib/gl3w")
        else()
            message(FATAL_ERROR "python 2.7/3.x not found.")
        endif()
    endif()

endif()

set(GL3W_SRC "${PROJECT_SOURCE_DIR}/extlib/gl3w/src/gl3w.c")
include_directories("${PROJECT_SOURCE_DIR}/extlib/gl3w/include")


# ----------------------------------------------------------------------------
# imgui

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extlib/imgui/imgui.h")
    execute_process(COMMAND "git clone https://github.com/ocornut/imgui.git"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib")
    execute_process(COMMAND "git checkout -b v1.67 refs/tags/v1.67"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib/imgui")
endif()

set(IMGUI_SRC
    "${PROJECT_SOURCE_DIR}/extlib/imgui/imgui.cpp"
    "${PROJECT_SOURCE_DIR}/extlib/imgui/imgui_draw.cpp"
    "${PROJECT_SOURCE_DIR}/extlib/imgui/imgui_demo.cpp"
    "${PROJECT_SOURCE_DIR}/extlib/imgui/imgui_widgets.cpp"
    "${PROJECT_SOURCE_DIR}/extlib/imgui/examples/imgui_impl_opengl3.cpp"
    "${PROJECT_SOURCE_DIR}/extlib/imgui/examples/imgui_impl_sdl.cpp"
)
include_directories("${PROJECT_SOURCE_DIR}/extlib/imgui")
include_directories("${PROJECT_SOURCE_DIR}/extlib/imgui/examples")

# ----------------------------------------------------------------------------
# fmtlib/fmt
if(EXISTS "${PROJECT_SOURCE_DIR}/extlib/fmt/include/fmt/format.hpp")
    add_definitions(-DFMT_HEADER_ONLY)
    include_directories("${PROJECT_SOURCE_DIR}/extlib/fmt/include")
else()
    execute_process(COMMAND "git clone https://github.com/fmtlib/fmt.git"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib")
    execute_process(COMMAND "git checkout -b v5.3.0 refs/tags/5.3.0"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib/fmt")

    add_definitions(-DFMT_HEADER_ONLY)
    include_directories("${PROJECT_SOURCE_DIR}/extlib/fmt/include")
endif()

# ----------------------------------------------------------------------------
# spray
include_directories(${PROJECT_SOURCE_DIR})
add_subdirectory(src)
