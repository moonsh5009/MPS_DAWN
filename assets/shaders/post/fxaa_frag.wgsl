@group(0) @binding(0) var color_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

const FXAA_REDUCE_MIN: f32 = 1.0 / 128.0;
const FXAA_REDUCE_MUL: f32 = 1.0 / 8.0;
const FXAA_SPAN_MAX: f32 = 8.0;

fn luminance(color: vec3f) -> f32 {
    return dot(color, vec3f(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let tex_size = vec2f(textureDimensions(color_texture));
    let inv_size = 1.0 / tex_size;

    let color_center = textureSample(color_texture, tex_sampler, uv).rgb;
    let color_tl = textureSample(color_texture, tex_sampler, uv + vec2f(-1.0, -1.0) * inv_size).rgb;
    let color_tr = textureSample(color_texture, tex_sampler, uv + vec2f( 1.0, -1.0) * inv_size).rgb;
    let color_bl = textureSample(color_texture, tex_sampler, uv + vec2f(-1.0,  1.0) * inv_size).rgb;
    let color_br = textureSample(color_texture, tex_sampler, uv + vec2f( 1.0,  1.0) * inv_size).rgb;

    let luma_center = luminance(color_center);
    let luma_tl = luminance(color_tl);
    let luma_tr = luminance(color_tr);
    let luma_bl = luminance(color_bl);
    let luma_br = luminance(color_br);

    let luma_min = min(luma_center, min(min(luma_tl, luma_tr), min(luma_bl, luma_br)));
    let luma_max = max(luma_center, max(max(luma_tl, luma_tr), max(luma_bl, luma_br)));

    let luma_range = luma_max - luma_min;

    // Skip FXAA if contrast is low
    if (luma_range < max(0.0312, luma_max * 0.125)) {
        return vec4f(color_center, 1.0);
    }

    var dir: vec2f;
    dir.x = -((luma_tl + luma_tr) - (luma_bl + luma_br));
    dir.y =  ((luma_tl + luma_bl) - (luma_tr + luma_br));

    let dir_reduce = max((luma_tl + luma_tr + luma_bl + luma_br) * 0.25 * FXAA_REDUCE_MUL, FXAA_REDUCE_MIN);
    let rcp_dir_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + dir_reduce);

    let clamped_dir = clamp(dir * rcp_dir_min, vec2f(-FXAA_SPAN_MAX), vec2f(FXAA_SPAN_MAX)) * inv_size;

    let result_a = 0.5 * (
        textureSample(color_texture, tex_sampler, uv + clamped_dir * (1.0 / 3.0 - 0.5)).rgb +
        textureSample(color_texture, tex_sampler, uv + clamped_dir * (2.0 / 3.0 - 0.5)).rgb
    );

    let result_b = result_a * 0.5 + 0.25 * (
        textureSample(color_texture, tex_sampler, uv + clamped_dir * -0.5).rgb +
        textureSample(color_texture, tex_sampler, uv + clamped_dir *  0.5).rgb
    );

    let luma_b = luminance(result_b);

    if (luma_b < luma_min || luma_b > luma_max) {
        return vec4f(result_a, 1.0);
    }
    return vec4f(result_b, 1.0);
}
