#include <spray/sdl/window.hpp>
#include <spray/sdl/texture.hpp>
#include <spray/sdl/resource.hpp>
#include <chrono>
#include <thread>

#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_opengl3.h>
#include <GL/gl3w.h>

int main()
{
    spray::sdl::resource sdl;
    spray::sdl::window<std::unique_ptr>  win("sample",  640, 480);
    spray::sdl::texture<std::unique_ptr> tex(win, SDL_PIXELFORMAT_RGBA32,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             640, 480);

    // ----------------------------------------------------------------------
    // set window color yellow
    {
        const auto tex_v = tex.lock<std::uint32_t>(SDL_Rect{0, 0, 640, 480});
        for(std::size_t x=0; x<640; ++x)
        {
            for(std::size_t y=0; y<480; ++y)
            {
                //                  RRGGBBAA
                tex_v.write(x, y, 0xFFFF00FF);
            }
        }
    }
    spray::sdl::render(win, tex);
    win.update();

    // ----------------------------------------------------------------------
    // preparation for OpenGL to use imgui
    const char* glsl_version = "#version 150";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_DisplayMode current;
    SDL_GetCurrentDisplayMode(0, &current);
    SDL_GLContext gl_context = SDL_GL_CreateContext(win.backend_window_ptr());
    SDL_GL_SetSwapInterval(1); // Enable vsync

    const auto gl3w_err = gl3wInit();
    if(gl3w_err != 0)
    {
        spray::log(spray::log_level::error, "failed to init OpenGL");
        return 1;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplSDL2_InitForOpenGL(win.backend_window_ptr(), gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);


    while(true)
    {
        SDL_Event ev;
        if(SDL_PollEvent(&ev))
        {
            ImGui_ImplSDL2_ProcessEvent(&ev);
            if(ev.type == SDL_WINDOWEVENT)
            {
                if(ev.window.event == SDL_WINDOWEVENT_CLOSE)
                {
                    break;
                }
            }
            if(ev.type == SDL_QUIT)
            {
                break;
            }
        }
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame(win.backend_window_ptr());
        ImGui::NewFrame();

        if (show_demo_window)
        {
            ImGui::ShowDemoWindow(&show_demo_window);
        }

        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!");

            ImGui::Text("This is some useful text.");
            ImGui::Checkbox("Demo Window", &show_demo_window);
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
            ImGui::ColorEdit3("clear color", (float*)&clear_color);

            if (ImGui::Button("Button"))
            {
                counter++;
            }
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window)
        {
            ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }

        // Rendering
        ImGui::Render();

        SDL_GL_MakeCurrent(win.backend_window_ptr(), gl_context);
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(win.backend_window_ptr());
    }
    return 0;
}
