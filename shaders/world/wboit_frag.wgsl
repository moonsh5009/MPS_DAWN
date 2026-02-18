#import "header/camera.wgsl"
#import "header/light.wgsl"

struct FragmentInput {
    @builtin(position) frag_coord: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
};

struct WBOITOutput {
    @location(0) accum: vec4f,
    @location(1) reveal: f32,
};

@fragment
fn fs_main(input: FragmentInput) -> WBOITOutput {
    let view_dir = normalize(u_camera.position.xyz - input.world_pos);
    let lit_color = blinn_phong(input.normal, view_dir, input.color.rgb);
    let alpha = input.color.a;

    // Weight function based on depth
    let depth = input.frag_coord.z;
    let weight = alpha * clamp(10.0 / (1e-5 + pow(depth / 10.0, 3.0) + pow(depth / 200.0, 6.0)), 1e-2, 3000.0);

    var output: WBOITOutput;
    output.accum = vec4f(lit_color * alpha * weight, alpha * weight);
    output.reveal = alpha;
    return output;
}
