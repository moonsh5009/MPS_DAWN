// CAS-based atomic float addition for storage buffers
// Used by force accumulation and diagonal Hessian assembly

fn atomicAddFloat(addr: ptr<storage, atomic<u32>, read_write>, val: f32) {
    var old_val = atomicLoad(addr);
    loop {
        let new_val = bitcast<u32>(bitcast<f32>(old_val) + val);
        let result = atomicCompareExchangeWeak(addr, old_val, new_val);
        if result.exchanged {
            break;
        }
        old_val = result.old_value;
    }
}
