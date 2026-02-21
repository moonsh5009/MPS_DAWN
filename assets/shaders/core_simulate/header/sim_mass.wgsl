// SimMass struct — layout-compatible with C++ SimMass.
// Storage buffer, std430: alignment 4, size 8, stride 8.

struct SimMass {
    mass: f32,
    inv_mass: f32,  // 0 = pinned
};
