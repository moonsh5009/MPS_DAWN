// Physics parameters uniform — layout-compatible with PhysicsParamsGPU C++ struct.
// Binding 0: global physics constants managed by DeviceDB singleton.
//
// Layout (32 bytes = 2 x vec4):
//   [0..3]   dt, gravity_x, gravity_y, gravity_z   (f32 x4)
//   [4..7]   damping, inv_dt, dt_sq, inv_dt_sq     (f32 x4)

struct PhysicsParams {
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    damping: f32,
    inv_dt: f32,
    dt_sq: f32,
    inv_dt_sq: f32,
};
