#import "header/camera.wgsl"
#import "header/light.wgsl"

struct FragmentInput {
    @location(0) world_pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
};

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4f {
    let view_dir = normalize(u_camera.position.xyz - input.world_pos);
    let lit_color = blinn_phong(input.normal, view_dir, input.color.rgb);
    return vec4f(lit_color, input.color.a);
}
