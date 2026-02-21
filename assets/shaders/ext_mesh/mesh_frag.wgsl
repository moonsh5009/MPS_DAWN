#import "header/camera.wgsl"
#import "header/light.wgsl"

struct FragmentInput {
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
    @builtin(front_facing) front_facing: bool,
};

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4f {
    let front_color = vec3f(0.4, 0.55, 0.8);
    let back_color  = vec3f(0.7, 0.45, 0.35);

    let view_dir = normalize(u_camera.position.xyz - input.world_position);

    // Two-sided lighting: flip normal for back faces
    var normal = normalize(input.world_normal);
    if (!input.front_facing) {
        normal = -normal;
    }

    let mesh_color = select(back_color, front_color, input.front_facing);
    let lit_color = blinn_phong(normal, view_dir, mesh_color);
    return vec4f(lit_color, 1.0);
}
