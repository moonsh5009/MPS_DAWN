add_library(core_render STATIC
    pipeline/render_pipeline_builder.cpp
    camera/camera.cpp
    camera/camera_controller.cpp
    uniform/camera_uniform.cpp
    uniform/light_uniform.cpp
    pass/render_pass_builder.cpp
    pass/render_encoder.cpp
    target/render_target.cpp
    geometry/draw_command.cpp
    post/fullscreen_quad.cpp
    post/fxaa_pass.cpp
    post/wboit_pass.cpp
    render_engine.cpp
)

target_include_directories(core_render PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(core_render PUBLIC
    mps::core_util
    mps::core_gpu
    mps::core_platform
)

add_library(mps::core_render ALIAS core_render)
