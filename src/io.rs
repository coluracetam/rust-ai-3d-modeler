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

/// Exports the visible parts as a binary glTF with named nodes, materials and colours.
pub fn export_glb_file(scene: &Scene, path: &Path) -> Result<(), String> {
    let parts: Vec<&crate::scene::SceneObject> = scene.objects.iter().filter(|o| o.visible).collect();
    crate::gltf::export_glb(&parts, path, &crate::gltf::GltfOptions::default()).map(|_| ())
}

/// Reads every mesh of a binary glTF into one mesh, applying node transforms and keeping
/// vertex colours. Godot collision-hint nodes (`-colonly`, `-convcolonly`) are skipped.
pub fn import_glb_file(path: &Path) -> Result<Mesh, String> {
    use glam::{Mat4, Quat};
    use serde_json::Value;
    let data = fs::read(path).map_err(|e| format!("Failed to read GLB: {e}"))?;
    if data.len() < 20 {
        return Err("File is too short to be a valid GLB".to_string());
    }
    if u32::from_le_bytes(data[0..4].try_into().unwrap()) != 0x46546C67 {
        return Err("Invalid GLB magic header".to_string());
    }
    let json_len = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let json_end = 20 + json_len;
    if data.len() < json_end + 8 {
        return Err("Truncated GLB chunks".to_string());
    }
    let doc: Value = serde_json::from_slice(&data[20..json_end])
        .map_err(|e| format!("Invalid GLB JSON chunk: {e}"))?;
    let bin_len = u32::from_le_bytes(data[json_end..json_end + 4].try_into().unwrap()) as usize;
    let bin_start = json_end + 8;
    if data.len() < bin_start + bin_len {
        return Err("Truncated GLB binary buffer".to_string());
    }
    let bin = &data[bin_start..bin_start + bin_len];
    let empty = Vec::new();
    let list = |key: &str| doc[key].as_array().unwrap_or(&empty);
    let (accessors, views, meshes, nodes, materials) = (
        list("accessors"),
        list("bufferViews"),
        list("meshes"),
        list("nodes"),
        list("materials"),
    );
    if meshes.is_empty() {
        return Err("No mesh in GLB".to_string());
    }

    // Reads accessor `index` as rows of `width` numbers (floats as-is, integers as values,
    // normalised integers scaled to the component's normalised range).
    let read = |index: &Value, width: usize| -> Result<Vec<f64>, String> {
        let accessor = index
            .as_u64()
            .and_then(|i| accessors.get(i as usize))
            .ok_or("GLB refers to a missing accessor")?;
        if !accessor["sparse"].is_null() {
            return Err("Sparse GLB accessors are unsupported".into());
        }
        let read_size = |value: &Value| value.as_u64().and_then(|n| usize::try_from(n).ok());
        let count = read_size(&accessor["count"]).filter(|&n| n > 0).ok_or("Invalid GLB accessor count")?;
        let component = accessor["componentType"].as_u64().unwrap_or(5126);
        let size = match component {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            other => return Err(format!("Unsupported GLB component type {other}")),
        };
        let view = read_size(&accessor["bufferView"])
            .and_then(|i| views.get(i))
            .ok_or("GLB accessor has no valid buffer view")?;
        if view["buffer"].as_u64().unwrap_or(0) != 0 {
            return Err("External GLB buffers are unsupported".into());
        }
        let view_offset = read_size(&view["byteOffset"]).unwrap_or(0);
        let view_bytes = read_size(&view["byteLength"])
            .and_then(|len| bin.get(view_offset..view_offset.checked_add(len)?))
            .ok_or("GLB buffer view reads past the binary chunk")?;
        let start = read_size(&accessor["byteOffset"]).unwrap_or(0);
        let row_size = size * width;
        let stride = read_size(&view["byteStride"]).unwrap_or(row_size);
        let end = (count - 1)
            .checked_mul(stride)
            .and_then(|last| last.checked_add(start))
            .and_then(|last| last.checked_add(row_size));
        if stride < row_size || end.is_none_or(|end| end > view_bytes.len()) {
            return Err("GLB accessor reads past its buffer view".into());
        }
        let normalized = accessor["normalized"].as_bool().unwrap_or(false);
        let mut out = Vec::with_capacity(count * width);
        for row in 0..count {
            for column in 0..width {
                let at = start + row * stride + column * size;
                let bytes = &view_bytes[at..at + size];
                let (value, max) = match component {
                    5120 => (bytes[0] as i8 as f64, 127.0),
                    5121 => (bytes[0] as f64, 255.0),
                    5122 => (i16::from_le_bytes(bytes.try_into().unwrap()) as f64, 32767.0),
                    5123 => (u16::from_le_bytes(bytes.try_into().unwrap()) as f64, 65535.0),
                    5125 => (u32::from_le_bytes(bytes.try_into().unwrap()) as f64, 1.0),
                    _ => (f32::from_le_bytes(bytes.try_into().unwrap()) as f64, 1.0),
                };
                out.push(if normalized && component != 5126 { (value / max).max(-1.0) } else { value });
            }
        }
        Ok(out)
    };

    let mut positions: Vec<Vec3> = Vec::new();
    let mut colors: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<Vec<usize>> = Vec::new();
    let mut add_mesh = |mesh: &Value, matrix: Mat4| -> Result<(), String> {
        for primitive in mesh["primitives"].as_array().unwrap_or(&empty) {
            if primitive["mode"].as_u64().unwrap_or(4) != 4 {
                continue;
            }
            let attributes = &primitive["attributes"];
            if attributes["POSITION"].is_null() {
                continue;
            }
            let base = positions.len();
            let points = read(&attributes["POSITION"], 3)?;
            positions.extend(points.chunks_exact(3).map(|p| {
                matrix.transform_point3(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32))
            }));
            let count = positions.len() - base;
            if positions[base..].iter().any(|p| !p.is_finite()) {
                return Err("GLB contains non-finite vertex positions".into());
            }
            let factor = primitive["material"].as_u64().and_then(|i| materials.get(i as usize));
            let factor = [0, 1, 2].map(|channel| {
                factor.and_then(|m| m["pbrMetallicRoughness"]["baseColorFactor"][channel].as_f64())
                    .unwrap_or(1.0).clamp(0.0, 1.0) as f32
            });
            let color_accessor = &attributes["COLOR_0"];
            if color_accessor.is_null() {
                colors.resize(positions.len(), factor);
            } else {
                let width = color_accessor
                    .as_u64()
                    .and_then(|i| accessors.get(i as usize))
                    .map_or(3, |a| if a["type"] == "VEC4" { 4 } else { 3 });
                let values = read(color_accessor, width)?;
                if values.len() / width != count {
                    return Err("GLB vertex colour count differs from the position count".into());
                }
                colors.extend(values.chunks_exact(width).map(|c| {
                    [0, 1, 2].map(|i| (c[i] as f32 * factor[i]).clamp(0.0, 1.0))
                }));
                if colors[base..].iter().flatten().any(|value| !value.is_finite()) {
                    return Err("GLB contains non-finite vertex colours".into());
                }
            }
            let indices: Vec<usize> = if primitive["indices"].is_null() {
                (0..count).collect()
            } else {
                read(&primitive["indices"], 1)?
                    .into_iter()
                    .map(|i| {
                        if i >= 0.0 && i < count as f64 && i.fract() == 0.0 {
                            Ok(i as usize)
                        } else {
                            Err("GLB contains an invalid vertex index")
                        }
                    })
                    .collect::<Result<_, _>>()?
            };
            let flip = matrix.determinant() < 0.0;
            for tri in indices.chunks_exact(3) {
                let (a, b, c) = (base + tri[0], base + tri[1], base + tri[2]);
                faces.push(if flip { vec![a, c, b] } else { vec![a, b, c] });
            }
        }
        Ok(())
    };

    if nodes.is_empty() {
        for mesh in meshes {
            add_mesh(mesh, Mat4::IDENTITY)?;
        }
    } else {
        let numbers = |v: &Value, n: usize| -> Option<Vec<f32>> {
            let a = v.as_array()?;
            (a.len() == n).then(|| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
        };
        let local = |node: &Value| -> Mat4 {
            if let Some(m) = numbers(&node["matrix"], 16) {
                return Mat4::from_cols_slice(&m);
            }
            let t = numbers(&node["translation"], 3).map_or(Vec3::ZERO, |v| Vec3::from_slice(&v));
            let r = numbers(&node["rotation"], 4).map_or(Quat::IDENTITY, |v| Quat::from_slice(&v).normalize());
            let s = numbers(&node["scale"], 3).map_or(Vec3::ONE, |v| Vec3::from_slice(&v));
            Mat4::from_scale_rotation_translation(s, r, t)
        };
        // Roots are the default scene's nodes, or every node that is nobody's child.
        let scene = doc["scene"].as_u64().unwrap_or(0) as usize;
        let mut stack: Vec<(usize, Mat4)> = match doc["scenes"][scene]["nodes"].as_array() {
            Some(roots) => roots.iter().filter_map(|r| Some((r.as_u64()? as usize, Mat4::IDENTITY))).collect(),
            None => {
                let children: std::collections::HashSet<u64> = nodes
                    .iter()
                    .flat_map(|n| n["children"].as_array().into_iter().flatten())
                    .filter_map(|c| c.as_u64())
                    .collect();
                (0..nodes.len()).filter(|i| !children.contains(&(*i as u64))).map(|i| (i, Mat4::IDENTITY)).collect()
            }
        };
        let mut visited = vec![false; nodes.len()];
        while let Some((index, parent)) = stack.pop() {
            let Some(node) = nodes.get(index) else { continue };
            if std::mem::replace(&mut visited[index], true) {
                continue;
            }
            let matrix = parent * local(node);
            let name = node["name"].as_str().unwrap_or("");
            let hint = ["-colonly", "-convcolonly"].iter().any(|s| name.ends_with(s));
            if let (false, Some(mesh)) = (hint, node["mesh"].as_u64().and_then(|m| meshes.get(m as usize))) {
                add_mesh(mesh, matrix)?;
            }
            for child in node["children"].as_array().into_iter().flatten() {
                if let Some(child) = child.as_u64() {
                    stack.push((child as usize, matrix));
                }
            }
        }
    }
    if positions.is_empty() || faces.is_empty() {
        return Err("The GLB has no triangle geometry".to_string());
    }
    let mut mesh = Mesh::new(positions, faces);
    mesh.colors = colors.into();
    Ok(mesh)
}
