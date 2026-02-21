struct Camera {
    view_mat: mat4x4f,
    view_inv_mat: mat4x4f,
    proj_mat: mat4x4f,
    proj_inv_mat: mat4x4f,
    position: vec4f,
    viewport: vec4f,
    frustum: vec2f,
    padding: vec2f,
};

@group(0) @binding(0) var<uniform> u_camera: Camera;
