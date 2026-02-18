struct Light {
    direction: vec4f,
    ambient: vec4f,
    diffuse: vec4f,
    specular: vec4f,
};

@group(0) @binding(1) var<uniform> u_light: Light;

fn blinn_phong(normal: vec3f, view_dir: vec3f, color: vec3f) -> vec3f {
    let n = normalize(normal);
    let light_dir = normalize(-u_light.direction.xyz);

    // Ambient
    let ambient = u_light.ambient.rgb * u_light.ambient.a;

    // Diffuse
    let diff = max(dot(n, light_dir), 0.0);
    let diffuse = u_light.diffuse.rgb * u_light.diffuse.a * diff;

    // Specular (Blinn-Phong)
    let half_dir = normalize(light_dir + view_dir);
    let spec = pow(max(dot(n, half_dir), 0.0), u_light.specular.a);
    let specular = u_light.specular.rgb * spec;

    return color * (ambient + diffuse) + specular;
}
