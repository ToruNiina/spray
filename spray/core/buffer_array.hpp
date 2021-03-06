#ifndef SPRAY_CORE_RENDER_BUFFER_ARRAY_HPP
#define SPRAY_CORE_RENDER_BUFFER_ARRAY_HPP
#include <glad/glad.h> // use OpenGL functions
#include <spray/util/cuda_assert.hpp>
#include <spray/util/scope_exit.hpp>
#include <cuda_gl_interop.h>
#include <utility>

namespace spray
{
namespace core
{

struct buffer_array
{
    buffer_array(const std::size_t width, const std::size_t height)
        : cuda_graphics_resource_(nullptr), cuda_array_(nullptr)
    {
        glCreateFramebuffers (1, std::addressof(this->frame_buffer_));
        glCreateRenderbuffers(1, std::addressof(this->render_buffer_));
        glNamedFramebufferRenderbuffer(this->frame_buffer_,
            GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, this->render_buffer_);

        this->resize(width, height);
    }
    buffer_array(const std::pair<std::size_t, std::size_t> size)
        : cuda_graphics_resource_(nullptr), cuda_array_(nullptr)
    {
        glCreateFramebuffers (1, std::addressof(this->frame_buffer_));
        glCreateRenderbuffers(1, std::addressof(this->render_buffer_));
        glNamedFramebufferRenderbuffer(this->frame_buffer_,
            GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, this->render_buffer_);

        this->resize(size);
    }

    buffer_array(const buffer_array&)            = delete;
    buffer_array& operator=(const buffer_array&) = delete;
    buffer_array(buffer_array&&)            = default;
    buffer_array& operator=(buffer_array&&) = default;
    ~buffer_array()
    {
        glDeleteFramebuffers (1, std::addressof(this->frame_buffer_));
        glDeleteRenderbuffers(1, std::addressof(this->render_buffer_));
        if(this->cuda_graphics_resource_ != nullptr)
        {
            cudaGraphicsUnregisterResource(this->cuda_graphics_resource_);
        }
    }

    void resize(const std::pair<std::size_t, std::size_t> size)
    {
        this->resize(size.first, size.second);
    }

    void resize(const std::size_t width, const std::size_t height)
    {
        this->width_  = width;
        this->height_ = height;

        // cleanup current resource
        if(this->cuda_graphics_resource_ != nullptr)
        {
            spray::util::cuda_assert(cudaGraphicsUnregisterResource(
                        this->cuda_graphics_resource_));
        }

        // resize Renderbuffer
        glNamedRenderbufferStorage(this->render_buffer_, GL_RGBA8, width, height);

        // make cuda Array to map the render buffer and edit it from cuda
        spray::util::cuda_assert(cudaGraphicsGLRegisterImage(
            std::addressof(this->cuda_graphics_resource_),
            this->render_buffer_, GL_RENDERBUFFER,
            cudaGraphicsRegisterFlagsSurfaceLoadStore |
            cudaGraphicsRegisterFlagsWriteDiscard));

        // lock graphics resource from OpenGL
        spray::util::cuda_assert(cudaGraphicsMapResources(
            1, std::addressof(this->cuda_graphics_resource_), 0));

        // unlock it when exit from this scope.
        const auto force_unlock = spray::util::make_scope_exit([this]() -> void {
                spray::util::cuda_assert(cudaGraphicsUnmapResources(
                    1, std::addressof(this->cuda_graphics_resource_), 0));
            });

        // bind graphics resource to cudaArray_t
        spray::util::cuda_assert(cudaGraphicsSubResourceGetMappedArray(
            std::addressof(this->cuda_array_), this->cuda_graphics_resource_,
            0, 0));

        // here UnmapResources is called, even if SubResourceGetMappedArray fail
        return;
    }

    std::size_t width()  const noexcept {return width_;}
    std::size_t height() const noexcept {return height_;}

    GLuint frame_buffer()  const noexcept {return frame_buffer_;}
    GLuint render_buffer() const noexcept {return render_buffer_;}

    cudaArray_const_t array() const noexcept {return cuda_array_;}

  private:

    std::size_t width_;    // stores current size (changed only by resize func)
    std::size_t height_;   // ditto
    GLuint frame_buffer_;  // OpenGL Framebuffer
    GLuint render_buffer_; // OpenGL Renderbuffer being bound to the Framebuffer
    cudaGraphicsResource_t cuda_graphics_resource_; // points to Renderbuffer
    cudaArray_t            cuda_array_;             // points to Renderbuffer
};

// XXX consider a more appropreate namespace. buffer_array is used for cuda,
//     but this function itself does not execute cuda stuff.
inline void blit_framebuffer(const buffer_array& bufarray)
{
    const int w = bufarray.width();
    const int h = bufarray.height();

    glBlitNamedFramebuffer(
        bufarray.frame_buffer(), /* default = */ 0,
        0, 0, w, h,
        0, h, w, 0,
        GL_COLOR_BUFFER_BIT, GL_NEAREST);
    return;
}

} // core
} // spray
#endif// SPRAY_CORE_RENDER_BUFFER_ARRAY_HPP
