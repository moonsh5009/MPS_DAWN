#import "header/camera.wgsl"
#import "header/light.wgsl"

struct FragmentInput {
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
};

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4f {
    // Soft blue cloth color
    let cloth_color = vec3f(0.4, 0.55, 0.8);

    let view_dir = normalize(u_camera.position.xyz - input.world_position);

    // Two-sided lighting: flip normal if facing away from camera
    var normal = input.world_normal;
    if (dot(normal, view_dir) < 0.0) {
        normal = -normal;
    }

    let lit_color = blinn_phong(normal, view_dir, cloth_color);
    return vec4f(lit_color, 1.0);
}
