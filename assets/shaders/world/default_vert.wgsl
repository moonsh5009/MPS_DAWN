#import "header/camera.wgsl"

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = vec4f(input.position, 1.0);
    output.position = u_camera.proj_mat * u_camera.view_mat * world_pos;
    output.world_pos = input.position;
    output.normal = input.normal;
    output.color = input.color;
    return output;
}
