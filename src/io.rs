use crate::mesh::Mesh;
use crate::scene::Scene;
use glam::Vec3;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

// =============================================================================
// WAVEFRONT OBJ IMPORT / EXPORT
// =============================================================================

pub fn export_obj_string(scene: &Scene) -> String {
    let mut out = String::from("# Exported by Rust 3D Modeler\n");
    let mut v_offset = 1usize;

    for obj in &scene.objects {
        if !obj.visible {
            continue;
        }
        out.push_str(&format!("\no {}\n", obj.name.replace(' ', "_")));
        let model = obj.transform.matrix();
        let normal_mat = model.inverse().transpose();

        for (i, &p) in obj.mesh.positions.iter().enumerate() {
            let wp = model.transform_point3(p);
            if let Some(&c) = obj.mesh.colors.get(i) {
                if (c[0] - 1.0).abs() > 1e-4
                    || (c[1] - 1.0).abs() > 1e-4
                    || (c[2] - 1.0).abs() > 1e-4
                {
                    out.push_str(&format!(
                        "v {:.6} {:.6} {:.6} {:.4} {:.4} {:.4}\n",
                        wp.x, wp.y, wp.z, c[0], c[1], c[2]
                    ));
                    continue;
                }
            }
            out.push_str(&format!("v {:.6} {:.6} {:.6}\n", wp.x, wp.y, wp.z));
        }

        for &n in &obj.mesh.normals {
            let wn = normal_mat.transform_vector3(n).normalize_or_zero();
            out.push_str(&format!("vn {:.6} {:.6} {:.6}\n", wn.x, wn.y, wn.z));
        }

        for face in &obj.mesh.faces {
            if face.len() < 3 {
                continue;
            }
            out.push('f');
            for &vi in face {
                let idx = vi + v_offset;
                out.push_str(&format!(" {}//{}", idx, idx));
            }
            out.push('\n');
        }

        v_offset += obj.mesh.positions.len();
    }

    out
}

pub fn export_obj_file(scene: &Scene, path: &Path) -> Result<(), String> {
    let content = export_obj_string(scene);
    fs::write(path, content).map_err(|e| format!("Failed to write OBJ: {e}"))
}

pub fn import_obj_str(content: &str) -> Result<Mesh, String> {
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    let mut has_vertex_colors = false;
    let mut faces = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(cmd) = parts.next() else {
            continue;
        };
        match cmd {
            "v" => {
                let coords: Vec<f32> = parts.take(6).filter_map(|s| s.parse().ok()).collect();
                if coords.len() >= 3 {
                    positions.push(Vec3::new(coords[0], coords[1], coords[2]));
                    if coords.len() >= 6 {
                        colors.push([coords[3], coords[4], coords[5]]);
                        has_vertex_colors = true;
                    } else {
                        colors.push([1.0, 1.0, 1.0]);
                    }
                }
            }
            "f" => {
                let mut face = Vec::new();
                for token in parts {
                    let v_str = token.split('/').next().unwrap_or("");
                    if let Ok(raw_idx) = v_str.parse::<isize>() {
                        let idx = if raw_idx > 0 {
                            (raw_idx as usize).saturating_sub(1)
                        } else {
                            (positions.len() as isize + raw_idx).max(0) as usize
                        };
                        if idx < positions.len() {
                            face.push(idx);
                        }
                    }
                }
                if face.len() >= 3 {
                    faces.push(face);
                }
            }
            _ => {}
        }
    }

    if positions.is_empty() {
        return Err("OBJ file contains no vertices".to_string());
    }

    let mut mesh = Mesh::new(positions, faces);
    if has_vertex_colors && colors.len() == mesh.positions.len() {
        mesh.colors = colors.into();
    }
    Ok(mesh)
}

pub fn import_obj_file(path: &Path) -> Result<Mesh, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("Failed to read OBJ: {e}"))?;
    import_obj_str(&content)
}

// =============================================================================
// STL IMPORT / EXPORT (BINARY & ASCII)
// =============================================================================

pub fn export_stl_file(scene: &Scene, path: &Path) -> Result<(), String> {
    let mut tris: Vec<([f32; 3], [Vec3; 3])> = Vec::new();

    for obj in &scene.objects {
        if !obj.visible {
            continue;
        }
        let model = obj.transform.matrix();
        for face in &obj.mesh.faces {
            if face.len() < 3 {
                continue;
            }
            let p0 = model.transform_point3(obj.mesh.positions[face[0]]);
            for i in 1..(face.len() - 1) {
                let p1 = model.transform_point3(obj.mesh.positions[face[i]]);
                let p2 = model.transform_point3(obj.mesh.positions[face[i + 1]]);
                let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
                tris.push((n.to_array(), [p0, p1, p2]));
            }
        }
    }

    let mut file =
        fs::File::create(path).map_err(|e| format!("Failed to create STL file: {e}"))?;
    let mut header = [0u8; 80];
    let title = b"Binary STL exported by Rust 3D Modeler";
    header[..title.len()].copy_from_slice(title);
    file.write_all(&header)
        .map_err(|e| format!("Failed to write STL header: {e}"))?;

    let count = tris.len() as u32;
    file.write_all(&count.to_le_bytes())
        .map_err(|e| format!("Failed to write STL count: {e}"))?;

    for (n, verts) in tris {
        for c in n {
            file.write_all(&c.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        for v in verts {
            for c in v.to_array() {
                file.write_all(&c.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }
        file.write_all(&0u16.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

pub fn import_stl_file(path: &Path) -> Result<Mesh, String> {
    let bytes = fs::read(path).map_err(|e| format!("Failed to read STL: {e}"))?;
    if bytes.len() < 15 {
        return Err("STL file is too small".to_string());
    }

    // Check if binary STL matches expected byte length: 84 + 50 * num_tris
    if bytes.len() >= 84 {
        let num_tris = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        if 84 + num_tris * 50 == bytes.len() {
            return parse_binary_stl(&bytes[84..], num_tris);
        }
    }

    // Fallback to ASCII STL
    let text = String::from_utf8_lossy(&bytes);
    parse_ascii_stl(&text)
}

fn weld_triangles_to_mesh(raw_tris: &[[Vec3; 3]]) -> Result<Mesh, String> {
    if raw_tris.is_empty() {
        return Err("STL file contains no triangles".to_string());
    }
    let mut positions = Vec::new();
    let mut faces = Vec::with_capacity(raw_tris.len());
    let mut weld_map: HashMap<(i64, i64, i64), usize> = HashMap::new();

    for tri in raw_tris {
        let mut face = Vec::with_capacity(3);
        for &v in tri {
            let key = (
                (v.x * 10000.0).round() as i64,
                (v.y * 10000.0).round() as i64,
                (v.z * 10000.0).round() as i64,
            );
            let idx = *weld_map.entry(key).or_insert_with(|| {
                let id = positions.len();
                positions.push(v);
                id
            });
            face.push(idx);
        }
        if face[0] != face[1] && face[1] != face[2] && face[2] != face[0] {
            faces.push(face);
        }
    }

    Ok(Mesh::new(positions, faces))
}

fn parse_binary_stl(data: &[u8], num_tris: usize) -> Result<Mesh, String> {
    let mut tris = Vec::with_capacity(num_tris);
    for i in 0..num_tris {
        let base = i * 50;
        let read_vec3 = |off: usize| -> Vec3 {
            let x = f32::from_le_bytes(data[off..off + 4].try_into().unwrap_or([0; 4]));
            let y = f32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap_or([0; 4]));
            let z = f32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap_or([0; 4]));
            Vec3::new(x, y, z)
        };
        let v0 = read_vec3(base + 12);
        let v1 = read_vec3(base + 24);
        let v2 = read_vec3(base + 36);
        tris.push([v0, v1, v2]);
    }
    weld_triangles_to_mesh(&tris)
}

fn parse_ascii_stl(text: &str) -> Result<Mesh, String> {
    let mut tris = Vec::new();
    let mut current = Vec::with_capacity(3);

    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("vertex") {
            let coords: Vec<f32> = rest
                .split_whitespace()
                .take(3)
                .filter_map(|s| s.parse().ok())
                .collect();
            if coords.len() == 3 {
                current.push(Vec3::new(coords[0], coords[1], coords[2]));
                if current.len() == 3 {
                    tris.push([current[0], current[1], current[2]]);
                    current.clear();
                }
            }
        }
    }

    weld_triangles_to_mesh(&tris)
}

// =============================================================================
// GLTF 2.0 BINARY (.GLB) IMPORT / EXPORT
// =============================================================================

/// Exports the visible scene into a standard Binary glTF 2.0 (`.glb`) file.
pub fn export_glb_file(scene: &Scene, path: &Path) -> Result<(), String> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let mut min_pos = Vec3::splat(f32::MAX);
    let mut max_pos = Vec3::splat(f32::MIN);

    for obj in &scene.objects {
        if !obj.visible {
            continue;
        }
        let model = obj.transform.matrix();
        let normal_mat = model.inverse().transpose();
        let base_v = positions.len() as u32;

        for (i, &p) in obj.mesh.positions.iter().enumerate() {
            let wp = model.transform_point3(p);
            min_pos = min_pos.min(wp);
            max_pos = max_pos.max(wp);
            positions.push(wp.to_array());

            let n = obj.mesh.normals.get(i).copied().unwrap_or(Vec3::Y);
            let wn = normal_mat.transform_vector3(n).normalize_or_zero();
            normals.push(wn.to_array());
        }

        for face in &obj.mesh.faces {
            if face.len() < 3 {
                continue;
            }
            let v0 = base_v + face[0] as u32;
            for i in 1..(face.len() - 1) {
                let v1 = base_v + face[i] as u32;
                let v2 = base_v + face[i + 1] as u32;
                indices.push(v0);
                indices.push(v1);
                indices.push(v2);
            }
        }
    }

    if positions.is_empty() || indices.is_empty() {
        return Err("Scene has no geometry to export".to_string());
    }

    let pos_bytes: &[u8] = bytemuck::cast_slice(&positions);
    let norm_bytes: &[u8] = bytemuck::cast_slice(&normals);
    let idx_bytes: &[u8] = bytemuck::cast_slice(&indices);

    let pos_offset = 0usize;
    let pos_len = pos_bytes.len();
    let norm_offset = pos_offset + pos_len;
    let norm_len = norm_bytes.len();
    let idx_offset = norm_offset + norm_len;
    let idx_len = idx_bytes.len();

    let mut bin_chunk = Vec::with_capacity(pos_len + norm_len + idx_len + 4);
    bin_chunk.extend_from_slice(pos_bytes);
    bin_chunk.extend_from_slice(norm_bytes);
    bin_chunk.extend_from_slice(idx_bytes);
    while bin_chunk.len() % 4 != 0 {
        bin_chunk.push(0);
    }

    let json_val = serde_json::json!({
        "asset": {
            "version": "2.0",
            "generator": "Rust 3D Modeler"
        },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0, "name": "ExportedScene" }],
        "meshes": [{
            "name": "SceneMesh",
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1
                },
                "indices": 2,
                "mode": 4
            }]
        }],
        "buffers": [{
            "byteLength": bin_chunk.len()
        }],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": pos_len,
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": norm_offset,
                "byteLength": norm_len,
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": idx_offset,
                "byteLength": idx_len,
                "target": 34963
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": positions.len(),
                "type": "VEC3",
                "min": [min_pos.x, min_pos.y, min_pos.z],
                "max": [max_pos.x, max_pos.y, max_pos.z]
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": normals.len(),
                "type": "VEC3"
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5125,
                "count": indices.len(),
                "type": "SCALAR"
            }
        ]
    });

    let mut json_bytes =
        serde_json::to_vec(&json_val).map_err(|e| format!("Failed to serialize glTF JSON: {e}"))?;
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }

    let total_length = 12 + 8 + json_bytes.len() + 8 + bin_chunk.len();
    let mut glb = Vec::with_capacity(total_length);

    // GLB Header (magic "glTF", version 2, total_length)
    glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total_length as u32).to_le_bytes());

    // JSON Chunk (length, type "JSON" = 0x4E4F534A)
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
    glb.extend_from_slice(&json_bytes);

    // BIN Chunk (length, type "BIN\0" = 0x004E4942)
    glb.extend_from_slice(&(bin_chunk.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
    glb.extend_from_slice(&bin_chunk);

    fs::write(path, glb).map_err(|e| format!("Failed to write GLB file: {e}"))
}

pub fn import_glb_file(path: &Path) -> Result<Mesh, String> {
    let data = fs::read(path).map_err(|e| format!("Failed to read GLB: {e}"))?;
    if data.len() < 20 {
        return Err("File is too short to be a valid GLB".to_string());
    }
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != 0x46546C67 {
        return Err("Invalid GLB magic header".to_string());
    }

    let json_len = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let json_end = 20 + json_len;
    if data.len() < json_end + 8 {
        return Err("Truncated GLB chunks".to_string());
    }
    let json_val: serde_json::Value = serde_json::from_slice(&data[20..json_end])
        .map_err(|e| format!("Invalid GLB JSON chunk: {e}"))?;

    let bin_len = u32::from_le_bytes(data[json_end..json_end + 4].try_into().unwrap()) as usize;
    let bin_start = json_end + 8;
    if data.len() < bin_start + bin_len {
        return Err("Truncated GLB binary buffer".to_string());
    }
    let bin = &data[bin_start..bin_start + bin_len];

    let accessors = json_val["accessors"]
        .as_array()
        .ok_or("Missing accessors in GLB")?;
    let buffer_views = json_val["bufferViews"]
        .as_array()
        .ok_or("Missing bufferViews in GLB")?;
    let meshes = json_val["meshes"]
        .as_array()
        .ok_or("Missing meshes in GLB")?;

    let prim = &meshes
        .first()
        .ok_or("No mesh in GLB")?["primitives"]
        .as_array()
        .ok_or("No primitives in GLB")?
        .first()
        .ok_or("Empty primitives")?;

    let pos_acc_idx = prim["attributes"]["POSITION"]
        .as_u64()
        .ok_or("Missing POSITION attribute")? as usize;
    let pos_acc = &accessors[pos_acc_idx];
    let pos_bv_idx = pos_acc["bufferView"].as_u64().unwrap_or(0) as usize;
    let pos_bv = &buffer_views[pos_bv_idx];
    let pos_offset = (pos_bv["byteOffset"].as_u64().unwrap_or(0)
        + pos_acc["byteOffset"].as_u64().unwrap_or(0)) as usize;
    let pos_count = pos_acc["count"].as_u64().unwrap_or(0) as usize;

    let mut positions = Vec::with_capacity(pos_count);
    for i in 0..pos_count {
        let off = pos_offset + i * 12;
        if off + 12 <= bin.len() {
            let x = f32::from_le_bytes(bin[off..off + 4].try_into().unwrap());
            let y = f32::from_le_bytes(bin[off + 4..off + 8].try_into().unwrap());
            let z = f32::from_le_bytes(bin[off + 8..off + 12].try_into().unwrap());
            positions.push(Vec3::new(x, y, z));
        }
    }

    let mut faces = Vec::new();
    if let Some(idx_acc_idx) = prim["indices"].as_u64() {
        let idx_acc = &accessors[idx_acc_idx as usize];
        let idx_bv = &buffer_views[idx_acc["bufferView"].as_u64().unwrap_or(0) as usize];
        let idx_offset = (idx_bv["byteOffset"].as_u64().unwrap_or(0)
            + idx_acc["byteOffset"].as_u64().unwrap_or(0)) as usize;
        let idx_count = idx_acc["count"].as_u64().unwrap_or(0) as usize;
        let comp_type = idx_acc["componentType"].as_u64().unwrap_or(5125);

        let mut raw_indices = Vec::with_capacity(idx_count);
        for i in 0..idx_count {
            match comp_type {
                5123 => {
                    // UNSIGNED_SHORT
                    let off = idx_offset + i * 2;
                    if off + 2 <= bin.len() {
                        raw_indices
                            .push(u16::from_le_bytes(bin[off..off + 2].try_into().unwrap()) as usize);
                    }
                }
                5125 => {
                    // UNSIGNED_INT
                    let off = idx_offset + i * 4;
                    if off + 4 <= bin.len() {
                        raw_indices
                            .push(u32::from_le_bytes(bin[off..off + 4].try_into().unwrap()) as usize);
                    }
                }
                _ => {}
            }
        }
        for tri in raw_indices.chunks_exact(3) {
            faces.push(vec![tri[0], tri[1], tri[2]]);
        }
    } else {
        for tri in (0..positions.len()).collect::<Vec<_>>().chunks_exact(3) {
            faces.push(vec![tri[0], tri[1], tri[2]]);
        }
    }

    Ok(Mesh::new(positions, faces))
}
