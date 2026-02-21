// Compute per-face normals and scatter (atomic add) to vertex normals
// Dispatch: ceil(face_count / 64) workgroups

#import "ext_mesh/header/normal_params.wgsl"

const FP_SCALE: f32 = 1048576.0; // 2^20

struct Face {
    n0: u32,
    n1: u32,
    n2: u32,
    pad: u32,
};

@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> faces: array<Face>;
@group(0) @binding(3) var<storage, read_write> normals: array<atomic<i32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let fid = gid.x;
    if (fid >= params.face_count) {
        return;
    }

    let face = faces[fid];
    let p0 = positions[face.n0].xyz;
    let p1 = positions[face.n1].xyz;
    let p2 = positions[face.n2].xyz;

    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let fn_vec = cross(e1, e2);

    let nx = i32(fn_vec.x * FP_SCALE);
    let ny = i32(fn_vec.y * FP_SCALE);
    let nz = i32(fn_vec.z * FP_SCALE);

    let base0 = face.n0 * 4u;
    atomicAdd(&normals[base0 + 0u], nx);
    atomicAdd(&normals[base0 + 1u], ny);
    atomicAdd(&normals[base0 + 2u], nz);

    let base1 = face.n1 * 4u;
    atomicAdd(&normals[base1 + 0u], nx);
    atomicAdd(&normals[base1 + 1u], ny);
    atomicAdd(&normals[base1 + 2u], nz);

    let base2 = face.n2 * 4u;
    atomicAdd(&normals[base2 + 0u], nx);
    atomicAdd(&normals[base2 + 1u], ny);
    atomicAdd(&normals[base2 + 2u], nz);
}
