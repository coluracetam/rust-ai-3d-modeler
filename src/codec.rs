//! Compact serde encodings for bulk geometry arrays.
//!
//! Large float and index arrays are written as `"<tag>:<base64(deflate(little-endian bytes))>"`
//! strings instead of JSON number arrays. That keeps project metadata readable JSON while making
//! geometry several times smaller and faster to parse. Every decoder also accepts the legacy
//! plain JSON arrays, so older projects keep loading.
use base64::Engine;
use flate2::{Compression, read::DeflateDecoder, write::DeflateEncoder};
use glam::Vec3;
use serde::{Deserialize, Deserializer, Serializer, de::Error};
use std::io::{Read, Write};

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

fn pack(tag: &str, bytes: &[u8]) -> String {
    let mut enc = DeflateEncoder::new(Vec::with_capacity(bytes.len() / 2), Compression::fast());
    enc.write_all(bytes).expect("in-memory write");
    let z = enc.finish().expect("in-memory write");
    let mut out = String::with_capacity(tag.len() + 1 + z.len() * 4 / 3 + 4);
    out.push_str(tag);
    out.push(':');
    B64.encode_string(&z, &mut out);
    out
}

fn unpack(tag: &str, s: &str) -> Result<Vec<u8>, String> {
    let body = s
        .strip_prefix(tag)
        .and_then(|r| r.strip_prefix(':'))
        .ok_or_else(|| format!("expected '{tag}:' encoded data"))?;
    let z = B64.decode(body).map_err(|e| e.to_string())?;
    let mut bytes = Vec::with_capacity(z.len() * 2);
    DeflateDecoder::new(&z[..])
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

fn floats_to_bytes(it: impl Iterator<Item = f32>, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 4);
    for f in it {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn bytes_to_floats(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err("truncated float data".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Either<T> {
    Packed(String),
    Plain(T),
}

/// `Vec<Vec3>` (positions, normals).
pub mod vec3s {
    use super::*;
    pub fn serialize<S: Serializer>(v: &[Vec3], s: S) -> Result<S::Ok, S::Error> {
        let bytes = floats_to_bytes(v.iter().flat_map(|p| p.to_array()), v.len() * 3);
        s.serialize_str(&pack("f32", &bytes))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Vec3>, D::Error> {
        match Either::<Vec<Vec3>>::deserialize(d)? {
            Either::Plain(v) => Ok(v),
            Either::Packed(s) => {
                let f = bytes_to_floats(&unpack("f32", &s).map_err(D::Error::custom)?)
                    .map_err(D::Error::custom)?;
                if f.len() % 3 != 0 {
                    return Err(D::Error::custom("vector data is not a multiple of 3"));
                }
                Ok(f.chunks_exact(3).map(Vec3::from_slice).collect())
            }
        }
    }
}

/// `Vec<[f32; 3]>` (vertex and voxel colors).
pub mod rgbs {
    use super::*;
    pub fn serialize<S: Serializer>(v: &[[f32; 3]], s: S) -> Result<S::Ok, S::Error> {
        let bytes = floats_to_bytes(v.iter().flatten().copied(), v.len() * 3);
        s.serialize_str(&pack("f32", &bytes))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<[f32; 3]>, D::Error> {
        match Either::<Vec<[f32; 3]>>::deserialize(d)? {
            Either::Plain(v) => Ok(v),
            Either::Packed(s) => {
                let f = bytes_to_floats(&unpack("f32", &s).map_err(D::Error::custom)?)
                    .map_err(D::Error::custom)?;
                if f.len() % 3 != 0 {
                    return Err(D::Error::custom("color data is not a multiple of 3"));
                }
                Ok(f.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
            }
        }
    }
}

/// `Vec<f32>` (voxel SDF).
pub mod f32s {
    use super::*;
    pub fn serialize<S: Serializer>(v: &[f32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&pack("f32", &floats_to_bytes(v.iter().copied(), v.len())))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
        match Either::<Vec<f32>>::deserialize(d)? {
            Either::Plain(v) => Ok(v),
            Either::Packed(s) => bytes_to_floats(&unpack("f32", &s).map_err(D::Error::custom)?)
                .map_err(D::Error::custom),
        }
    }
}

/// `Vec<Vec<usize>>` polygon lists, stored as `[count, i0, i1, …]` u32 runs.
pub mod faces {
    use super::*;
    pub fn serialize<S: Serializer>(v: &[Vec<usize>], s: S) -> Result<S::Ok, S::Error> {
        let mut bytes = Vec::with_capacity(v.len() * 20);
        for f in v {
            bytes.extend_from_slice(&(f.len() as u32).to_le_bytes());
            for &i in f {
                bytes.extend_from_slice(&(i as u32).to_le_bytes());
            }
        }
        s.serialize_str(&pack("poly", &bytes))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Vec<usize>>, D::Error> {
        match Either::<Vec<Vec<usize>>>::deserialize(d)? {
            Either::Plain(v) => Ok(v),
            Either::Packed(s) => {
                let bytes = unpack("poly", &s).map_err(D::Error::custom)?;
                if bytes.len() % 4 != 0 {
                    return Err(D::Error::custom("truncated face data"));
                }
                let words: Vec<u32> = bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let mut out = Vec::new();
                let mut i = 0;
                while i < words.len() {
                    let n = words[i] as usize;
                    let end = i + 1 + n;
                    if end > words.len() {
                        return Err(D::Error::custom("truncated face data"));
                    }
                    out.push(words[i + 1..end].iter().map(|&x| x as usize).collect());
                    i = end;
                }
                Ok(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct T {
        #[serde(with = "vec3s")]
        p: Vec<Vec3>,
        #[serde(with = "rgbs")]
        c: Vec<[f32; 3]>,
        #[serde(with = "f32s")]
        s: Vec<f32>,
        #[serde(with = "faces")]
        f: Vec<Vec<usize>>,
    }

    #[test]
    fn packed_round_trip_and_legacy_arrays() {
        let t = T {
            p: vec![Vec3::new(1.5, -2.0, 3.25), Vec3::ZERO],
            c: vec![[0.1, 0.2, 0.3]],
            s: vec![-1.0, 0.5, f32::MAX],
            f: vec![vec![0, 1, 2], vec![2, 3, 4, 5], vec![]],
        };
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains("\"poly:"));
        assert_eq!(serde_json::from_str::<T>(&json).unwrap(), t);
        let legacy = r#"{"p":[[1.5,-2.0,3.25],[0,0,0]],"c":[[0.1,0.2,0.3]],"s":[-1.0,0.5,3.4028235e38],"f":[[0,1,2],[2,3,4,5],[]]}"#;
        assert_eq!(serde_json::from_str::<T>(legacy).unwrap(), t);
    }
}
