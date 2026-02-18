#import "header/camera.wgsl"

struct VertexInput {
    @location(0) position: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = u_camera.proj_mat * u_camera.view_mat * vec4f(input.position, 1.0);
    return output;
}
