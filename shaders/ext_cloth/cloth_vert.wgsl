#import "header/camera.wgsl"

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = vec4f(input.position, 1.0);
    output.clip_position = u_camera.proj_mat * u_camera.view_mat * world_pos;
    output.world_normal = input.normal;
    output.world_position = input.position;
    return output;
}
