#include <spray/core/world.cuh>
#include <imgui.h>
#include <algorithm>

namespace spray
{
namespace core
{

std::unique_ptr<world_base> make_world()
{
    return std::make_unique<world>();
}

bool world::update_gui()
{
    std::vector<std::size_t> remove_list;
    bool is_focused = false;
    for(const auto idx : this->sub_windows_)
    {
        if(idx == 0xFFFFFFFF)
        {
            ImGui::Begin("Background");
            is_focused = is_focused || ImGui::IsWindowFocused();
            ImGui::ColorEdit4("background color",
                    reinterpret_cast<float*>(std::addressof(this->bg_)));
            if (ImGui::Button("Close"))
            {
                remove_list.push_back(idx);
            }
            ImGui::End();
        }
        else
        {
            const auto name = std::string("particle#") + std::to_string(idx);
            ImGui::Begin(name.c_str());
            is_focused = is_focused || ImGui::IsWindowFocused();

            auto& buf = this->material_buffer_.at(idx);
            ImGui::ColorEdit3("albedo",
                    reinterpret_cast<float*>(std::addressof(buf.albedo)));
            ImGui::ColorEdit3("emission",
                    reinterpret_cast<float*>(std::addressof(buf.emission)));
            if(ImGui::Button("Apply changes"))
            {
                this->change_material_at(idx, buf);
            }
            if (ImGui::Button("Close"))
            {
                remove_list.push_back(idx);
            }
            ImGui::End();
        }
    }
    for(const auto rm : remove_list)
    {
        const auto pos = std::find(sub_windows_.begin(), sub_windows_.end(), rm);
        sub_windows_.erase(pos);
    }
    return is_focused;
}

} // core
} // spray
