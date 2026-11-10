struct GlobalUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    clip_plane: vec4<f32>,
};

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    // x = roughness, y = metallic, z = shading_mode (0 = Studio Lit, 1 = Sculpt Clay, 2 = Unlit Line), w = pattern_id
    params: vec4<f32>,
    // xyz = secondary_color, w = texture_scale
    tex_params: vec4<f32>,
    // xyz = emission_color, w = emission_strength
    emission: vec4<f32>,
    inspection: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> global: GlobalUniforms;

@group(1) @binding(0)
var<uniform> object: ObjectUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) local_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_p = object.model * vec4<f32>(in.position, 1.0);
    out.world_pos = world_p.xyz;
    out.local_pos = in.position;
    out.clip_position = global.view_proj * world_p;
    out.world_normal = normalize((object.normal_mat * vec4<f32>(in.normal, 0.0)).xyz);
    out.color = in.color;
    return out;
}

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);

    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);

    return mix(nxy0, nxy1, u.z);
}

fn fbm3(p: vec3<f32>) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var pos = p;
    for (var i = 0; i < 4; i = i + 1) {
        val += amp * noise3d(pos);
        pos = pos * 2.03 + vec3<f32>(13.7, 7.3, 19.1);
        amp *= 0.5;
    }
    return val;
}

// Tri-planar 2D UV coordinate selector based on dominant surface normal axis
fn triplanar_uv(pos: vec3<f32>, norm: vec3<f32>) -> vec2<f32> {
    let an = abs(norm);
    if (an.y > an.x && an.y > an.z) {
        return pos.xz;
    } else if (an.x > an.z) {
        return pos.zy;
    } else {
        return pos.xy;
    }
}

fn evaluate_procedural_pattern(
    pattern_id: i32,
    base_col: vec3<f32>,
    sec_col: vec3<f32>,
    local_pos: vec3<f32>,
    world_norm: vec3<f32>,
    scale: f32
) -> vec3<f32> {
    if (pattern_id <= 0) {
        return base_col;
    }

    let p = local_pos * max(scale, 0.1);
    let uv = triplanar_uv(p, world_norm);

    // 1: Checkerboard Tiles
    if (pattern_id == 1) {
        let cell = floor(p.x) + floor(p.y * 0.5) + floor(p.z);
        let is_alt = fract(cell * 0.5) * 2.0 > 0.5;
        let bevel = smoothstep(0.0, 0.06, min(fract(uv.x), 1.0 - fract(uv.x)))
                  * smoothstep(0.0, 0.06, min(fract(uv.y), 1.0 - fract(uv.y)));
        let tile_col = select(base_col, sec_col, is_alt);
        return mix(sec_col * 0.45, tile_col, bevel);
    }

    // 2: Running-Bond Masonry Bricks
    if (pattern_id == 2) {
        let row = floor(uv.y * 2.0);
        let brick_uv = vec2<f32>(uv.x + row * 0.5, uv.y * 2.0);
        let cell_id = floor(brick_uv);
        let f = fract(brick_uv);
        let mortar_x = smoothstep(0.0, 0.06, f.x) * smoothstep(0.0, 0.06, 1.0 - f.x);
        let mortar_y = smoothstep(0.0, 0.09, f.y) * smoothstep(0.0, 0.09, 1.0 - f.y);
        let is_brick = mortar_x * mortar_y;
        let variation = (hash31(vec3<f32>(cell_id.x, cell_id.y, row)) - 0.5) * 0.18;
        let grain = (fbm3(p * 3.0) - 0.5) * 0.10;
        let brick_col = clamp(base_col + vec3<f32>(variation + grain), vec3<f32>(0.0), vec3<f32>(1.0));
        return mix(sec_col, brick_col, is_brick);
    }

    // 3: Ashlar Castle Stone Blocks
    if (pattern_id == 3) {
        let row = floor(uv.y * 1.4);
        let jitter = hash31(vec3<f32>(row, 1.0, 2.0)) * 0.7;
        let st = vec2<f32>(uv.x * 0.9 + jitter, uv.y * 1.4);
        let id = floor(st);
        let f = fract(st);
        let n_warp = (fbm3(p * 2.2) - 0.5) * 0.07;
        let gx = smoothstep(0.0, 0.07, abs(f.x - 0.5 + n_warp));
        let edge_x = smoothstep(0.0, 0.06, min(f.x, 1.0 - f.x));
        let edge_y = smoothstep(0.0, 0.07, min(f.y, 1.0 - f.y));
        let seam = edge_x * edge_y;
        let tone = (hash31(vec3<f32>(id.x, id.y, 7.0)) - 0.5) * 0.16 + (fbm3(p * 4.0) - 0.5) * 0.12;
        let stone = clamp(base_col + vec3<f32>(tone), vec3<f32>(0.0), vec3<f32>(1.0));
        return mix(sec_col, stone, seam * gx);
    }

    // 4: Procedural 3D Wood Grain
    if (pattern_id == 4) {
        let warp = fbm3(p * 1.5) * 0.65;
        let radial = length(p.xz + vec2<f32>(warp)) * 4.5;
        let rings = smoothstep(-0.6, 0.6, sin(radial * 6.28318));
        let fibers = (noise3d(vec3<f32>(p.x * 12.0, p.y * 2.5, p.z * 12.0)) - 0.5) * 0.12;
        return clamp(mix(base_col, sec_col, rings * 0.75) + vec3<f32>(fibers), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // 5: Veined Marble
    if (pattern_id == 5) {
        let turb = fbm3(p * 1.8) * 5.5;
        let vein = pow(abs(sin(p.x * 1.5 + p.y * 1.2 + p.z * 0.8 + turb)), 0.35);
        return mix(sec_col, base_col, vein);
    }

    // 6: Arcane Hexagonal Grid
    if (pattern_id == 6) {
        let s = vec2<f32>(1.0, 1.7320508);
        let u = uv * 2.2;
        let a = fract(u / s) - 0.5;
        let b = fract((u - s * 0.5) / s) - 0.5;
        let ga = a * s;
        let gb = b * s;
        let g = select(gb, ga, dot(ga, ga) < dot(gb, gb));
        let hex_d = max(abs(g.x), dot(abs(g), vec2<f32>(0.5, 0.8660254)));
        let border = smoothstep(0.43, 0.48, hex_d);
        return mix(base_col, sec_col, border);
    }

    // 7: Rune Circuit Lines
    if (pattern_id == 7) {
        let grid_uv = fract(uv * 2.0);
        let cell = floor(uv * 2.0);
        let flip = hash31(vec3<f32>(cell.x, cell.y, 3.0)) > 0.5;
        let gx = select(grid_uv.x, 1.0 - grid_uv.x, flip);
        let d1 = abs(gx + grid_uv.y - 0.5);
        let d2 = abs(gx + grid_uv.y - 1.5);
        let line_d = min(d1, d2);
        let trace = 1.0 - smoothstep(0.03, 0.08, line_d);
        return mix(base_col, sec_col, trace);
    }

    return base_col;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (dot(global.clip_plane.xyz, in.world_pos) + global.clip_plane.w < 0.0) { discard; }
    let mode = i32(round(object.params.z));
    if (mode == 2) {
        return vec4<f32>(in.color, 1.0);
    }

    let N = normalize(in.world_normal);
    let V = normalize(global.camera_pos.xyz - in.world_pos);
    let pattern_id = i32(round(object.params.w));
    let surface_color = evaluate_procedural_pattern(
        pattern_id,
        in.color,
        object.tex_params.xyz,
        in.local_pos,
        N,
        object.tex_params.w
    );
    let emissive = object.emission.xyz * object.emission.w;

    if (mode == 1) {
        // Sculpt Clay / Studio MatCap-style shading
        let clay_base = mix(surface_color, vec3<f32>(0.84, 0.56, 0.44), 0.55);
        let key_dir = normalize(vec3<f32>(0.5, 0.8, 0.6));
        let fill_dir = normalize(vec3<f32>(-0.7, 0.3, -0.4));
        let ndotl = max(dot(N, key_dir), 0.0);
        let fill = max(dot(N, fill_dir), 0.0) * 0.35;
        let rim = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.28;
        let H = normalize(key_dir + V);
        let spec = pow(max(dot(N, H), 0.0), 28.0) * 0.18;
        let ambient = mix(vec3<f32>(0.18, 0.14, 0.13), vec3<f32>(0.34, 0.28, 0.25), N.y * 0.5 + 0.5);
        let col = clay_base * (ambient + vec3<f32>(0.85, 0.82, 0.78) * ndotl + fill) + vec3<f32>(spec + rim) + emissive;
        return vec4<f32>(col, object.inspection.x);
    }

    // Mode 0: Studio 3-Point PBR-inspired lighting + Procedural Texture + Emissive Glow
    let roughness = clamp(object.params.x, 0.05, 1.0);
    let metallic = clamp(object.params.y, 0.0, 1.0);

    let key_dir = normalize(vec3<f32>(0.6, 0.85, 0.5));
    let fill_dir = normalize(vec3<f32>(-0.75, 0.25, 0.4));
    let back_dir = normalize(vec3<f32>(0.1, 0.4, -0.9));

    let k_diff = max(dot(N, key_dir), 0.0);
    let f_diff = max(dot(N, fill_dir), 0.0) * 0.35;
    let b_diff = max(dot(N, back_dir), 0.0) * 0.25;

    let H = normalize(key_dir + V);
    let shininess = mix(96.0, 8.0, roughness);
    let spec_strength = mix(0.08, 0.65, 1.0 - roughness) + metallic * 0.35;
    let spec = pow(max(dot(N, H), 0.0), shininess) * spec_strength;

    let hemi = mix(vec3<f32>(0.14, 0.15, 0.18), vec3<f32>(0.30, 0.32, 0.36), N.y * 0.5 + 0.5);
    let diffuse_light = hemi + vec3<f32>(0.92, 0.90, 0.86) * k_diff + vec3<f32>(0.65, 0.75, 0.90) * f_diff + vec3<f32>(0.8, 0.8, 0.9) * b_diff;
    let spec_color = mix(vec3<f32>(1.0), surface_color, metallic);

    let final_rgb = surface_color * diffuse_light * (1.0 - metallic * 0.45) + spec_color * spec + emissive;
    return vec4<f32>(final_rgb, object.inspection.x);
}
