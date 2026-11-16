//! Image layers on parts: a title on a book spine, a label, a sign, a painting. A layer is an
//! image projected straight onto the part from a rectangle in its local space, so it needs no
//! UV unwrap, survives mesh edits and costs no triangles. The viewport shader, the texture
//! baker and the tests all evaluate a layer the same way (`TextureLayer::sample`).
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Layers per part. The viewport shader has this many slots.
pub const MAX_LAYERS: usize = 4;
/// Longest side of a layer image, in pixels.
pub const MAX_IMAGE_SIDE: u32 = 2048;

/// Straight-alpha sRGB pixels, row by row from the top.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl TextureImage {
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, String> {
        if width == 0 || height == 0 || width > MAX_IMAGE_SIDE || height > MAX_IMAGE_SIDE {
            return Err(format!("Images must be between 1 and {MAX_IMAGE_SIDE} pixels on each side"));
        }
        if rgba.len() != (width * height * 4) as usize {
            return Err("Image data does not match its size".into());
        }
        Ok(Self { width, height, rgba })
    }

    /// Reads a PNG file, shrinking it to `MAX_IMAGE_SIDE` if it is larger.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let image = image::open(path).map_err(|e| format!("Could not read {} as a PNG image: {e}", path.display()))?;
        let image = if image.width().max(image.height()) > MAX_IMAGE_SIDE {
            image.thumbnail(MAX_IMAGE_SIDE, MAX_IMAGE_SIDE)
        } else {
            image
        };
        let rgba = image.into_rgba8();
        Self::new(rgba.width(), rgba.height(), rgba.into_raw())
    }

    pub fn to_png(&self) -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        image::write_buffer_with_format(
            &mut std::io::Cursor::new(&mut bytes),
            &self.rgba,
            self.width,
            self.height,
            image::ExtendedColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .map_err(|e| e.to_string())?;
        Ok(bytes)
    }

    pub fn from_png(bytes: &[u8]) -> Result<Self, String> {
        let rgba = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)
            .map_err(|e| e.to_string())?
            .into_rgba8();
        Self::new(rgba.width(), rgba.height(), rgba.into_raw())
    }

    /// Bilinear sample at `u`, `v` in 0..1 (0, 0 = top-left): linear RGB and alpha.
    pub fn sample(&self, u: f32, v: f32) -> [f32; 4] {
        let x = (u * self.width as f32 - 0.5).clamp(0.0, self.width as f32 - 1.0);
        let y = (v * self.height as f32 - 0.5).clamp(0.0, self.height as f32 - 1.0);
        let (x0, y0) = (x.floor() as u32, y.floor() as u32);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let texel = |px: u32, py: u32| {
            let i = ((py * self.width + px) * 4) as usize;
            [
                srgb_to_linear(self.rgba[i]),
                srgb_to_linear(self.rgba[i + 1]),
                srgb_to_linear(self.rgba[i + 2]),
                self.rgba[i + 3] as f32 / 255.0,
            ]
        };
        let (a, b, c, d) = (texel(x0, y0), texel(x1, y0), texel(x0, y1), texel(x1, y1));
        [0, 1, 2, 3].map(|k| {
            (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy
        })
    }
}

pub fn srgb_to_linear(value: u8) -> f32 {
    let c = value as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0).round() as u8
}

/// Stored in project and asset files as `"png:<base64>"`.
mod png_text {
    use super::TextureImage;
    use base64::Engine;
    use serde::{Deserialize, Deserializer, Serializer};
    use std::sync::Arc;

    pub fn serialize<S: Serializer>(image: &Arc<TextureImage>, s: S) -> Result<S::Ok, S::Error> {
        let png = image.to_png().map_err(serde::ser::Error::custom)?;
        s.serialize_str(&format!("png:{}", base64::engine::general_purpose::STANDARD.encode(png)))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Arc<TextureImage>, D::Error> {
        let text = String::deserialize(d)?;
        let data = text.strip_prefix("png:").ok_or_else(|| serde::de::Error::custom("expected a png: image"))?;
        let png = base64::engine::general_purpose::STANDARD.decode(data).map_err(serde::de::Error::custom)?;
        TextureImage::from_png(&png).map(Arc::new).map_err(serde::de::Error::custom)
    }
}

/// An image projected onto a part. The image's top-left corner sits at `origin` (part-local
/// space), its top edge runs along `u_axis` and its left edge along `v_axis`; both vectors
/// carry the image's size. It lands on faces that look back along the projection (outward
/// normal `v_axis x u_axis`) and lie within `depth` of the image plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextureLayer {
    pub name: String,
    #[serde(with = "png_text")]
    pub image: Arc<TextureImage>,
    pub origin: Vec3,
    pub u_axis: Vec3,
    pub v_axis: Vec3,
    pub depth: f32,
    pub opacity: f32,
}

/// How squarely a face must look at the image to receive it (cosine of the angle).
pub const FACING: f32 = 0.3;

impl TextureLayer {
    pub fn normal(&self) -> Vec3 {
        self.v_axis.cross(self.u_axis).normalize_or_zero()
    }

    /// Vector dot products with these axes recover the image coordinates.
    pub fn projection_axes(&self, normal: Vec3) -> (Vec3, Vec3) {
        let u = normal.cross(self.v_axis);
        let v = self.u_axis.cross(normal);
        (u / u.dot(self.u_axis), v / v.dot(self.v_axis))
    }

    pub fn validate(&self) -> Result<(), String> {
        let finite = self.origin.is_finite() && self.u_axis.is_finite() && self.v_axis.is_finite();
        let normal = self.normal();
        let (u, v) = self.projection_axes(normal);
        if !finite || self.u_axis.length() < 1e-6 || self.v_axis.length() < 1e-6
            || normal == Vec3::ZERO || !u.is_finite() || !v.is_finite()
        {
            return Err(format!("Texture layer '{}' has no usable rectangle", self.name));
        }
        if !(self.depth.is_finite() && self.depth > 0.0 && self.opacity.is_finite()) {
            return Err(format!("Texture layer '{}' has an invalid depth or opacity", self.name));
        }
        let image = &self.image;
        if image.width == 0
            || image.height == 0
            || image.width.max(image.height) > MAX_IMAGE_SIDE
            || image.rgba.len() != (image.width * image.height * 4) as usize
        {
            return Err(format!("Texture layer '{}' has an invalid image", self.name));
        }
        Ok(())
    }

    /// Image coordinates of a local point, when the layer covers it on a face with `normal`.
    pub fn uv(&self, point: Vec3, normal: Vec3) -> Option<(f32, f32)> {
        let towards = self.normal();
        let offset = point - self.origin;
        if normal.dot(towards) < FACING || offset.dot(towards).abs() > self.depth {
            return None;
        }
        let (u_axis, v_axis) = self.projection_axes(towards);
        let u = offset.dot(u_axis);
        let v = offset.dot(v_axis);
        ((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)).then_some((u, v))
    }

    /// Linear RGB and coverage (alpha times opacity) at a local point, if the layer is there.
    pub fn sample(&self, point: Vec3, normal: Vec3) -> Option<[f32; 4]> {
        let (u, v) = self.uv(point, normal)?;
        let [r, g, b, a] = self.image.sample(u, v);
        Some([r, g, b, a * self.opacity.clamp(0.0, 1.0)])
    }

    /// Length of one image pixel on the surface, in local units.
    pub fn texel_size(&self) -> f32 {
        (self.u_axis.length() / self.image.width as f32).min(self.v_axis.length() / self.image.height as f32)
    }
}

/// The colour of a surface point once every layer is laid over `base`, first layer lowest.
pub fn apply_layers(layers: &[TextureLayer], base: [f32; 3], point: Vec3, normal: Vec3) -> [f32; 3] {
    layers.iter().fold(base, |color, layer| match layer.sample(point, normal) {
        Some([r, g, b, a]) => [
            color[0] + (r - color[0]) * a,
            color[1] + (g - color[1]) * a,
            color[2] + (b - color[2]) * a,
        ],
        None => color,
    })
}

// ---------------------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------------------

pub struct TextStyle {
    /// Linear RGB of the letters.
    pub color: [f32; 3],
    /// Linear RGB behind the letters; `None` leaves it transparent so the part shows through.
    pub background: Option<[f32; 3]>,
    pub bold: bool,
    /// Empty border as a fraction of the shorter image side.
    pub padding: f32,
}

/// Draws `text` as large as fits a `width` x `height` image, centred. `\n` starts a new line.
pub fn render_text(text: &str, width: u32, height: u32, style: &TextStyle) -> Result<TextureImage, String> {
    use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if lines.is_empty() {
        return Err("The text is empty".into());
    }
    let image = TextureImage::new(width, height, vec![0; (width * height * 4) as usize])?;
    let font_bytes: &[u8] = if style.bold {
        include_bytes!("../assets/fonts/FiraSans-SemiBold.ttf")
    } else {
        include_bytes!("../assets/fonts/FiraSans-Regular.ttf")
    };
    let font = FontRef::try_from_slice(font_bytes).map_err(|e| e.to_string())?;

    // Measure at a reference size, then scale so the widest line and the stack both fit.
    let line_width = |line: &str, scale: f32| -> f32 {
        let scaled = font.as_scaled(PxScale::from(scale));
        let mut width = 0.0;
        let mut previous = None;
        for c in line.chars() {
            let id = font.glyph_id(c);
            if let Some(p) = previous {
                width += scaled.kern(p, id);
            }
            width += scaled.h_advance(id);
            previous = Some(id);
        }
        width
    };
    let reference = 100.0_f32;
    let widest = lines.iter().map(|l| line_width(l, reference)).fold(1.0_f32, f32::max);
    let line_height = {
        let scaled = font.as_scaled(PxScale::from(reference));
        scaled.height() + scaled.line_gap()
    };
    let pad = style.padding.clamp(0.0, 0.45) * width.min(height) as f32;
    let (room_w, room_h) = ((width as f32 - 2.0 * pad).max(1.0), (height as f32 - 2.0 * pad).max(1.0));
    let scale = reference * (room_w / widest).min(room_h / (line_height * lines.len() as f32));
    let scaled = font.as_scaled(PxScale::from(scale));
    let step = scaled.height() + scaled.line_gap();
    let top = (height as f32 - step * lines.len() as f32) * 0.5 + scaled.line_gap() * 0.5;

    let mut coverage = vec![0.0_f32; (width * height) as usize];
    for (row, line) in lines.iter().enumerate() {
        let mut x = (width as f32 - line_width(line, scale)) * 0.5;
        let baseline = top + step * row as f32 + scaled.ascent();
        let mut previous = None;
        for c in line.chars() {
            let id = font.glyph_id(c);
            if let Some(p) = previous {
                x += scaled.kern(p, id);
            }
            let glyph = id.with_scale_and_position(PxScale::from(scale), ab_glyph::point(x, baseline));
            if let Some(outline) = font.outline_glyph(glyph) {
                let bounds = outline.px_bounds();
                outline.draw(|gx, gy, c| {
                    let (px, py) = (bounds.min.x as i32 + gx as i32, bounds.min.y as i32 + gy as i32);
                    if px >= 0 && py >= 0 && (px as u32) < width && (py as u32) < height {
                        let slot = &mut coverage[(py as u32 * width + px as u32) as usize];
                        *slot = slot.max(c);
                    }
                });
            }
            x += scaled.h_advance(id);
            previous = Some(id);
        }
    }

    let mut image = image;
    let ink = style.color.map(linear_to_srgb8);
    for (i, &cover) in coverage.iter().enumerate() {
        let pixel = &mut image.rgba[i * 4..i * 4 + 4];
        match style.background {
            Some(background) => {
                // Blend in linear light so thin strokes keep their weight.
                let mixed = [0, 1, 2].map(|k| background[k] + (style.color[k] - background[k]) * cover);
                pixel[..3].copy_from_slice(&mixed.map(linear_to_srgb8));
                pixel[3] = 255;
            }
            None => {
                pixel[..3].copy_from_slice(&ink);
                pixel[3] = (cover * 255.0).round() as u8;
            }
        }
    }
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker() -> Arc<TextureImage> {
        // 2x2: red, green / blue, transparent.
        Arc::new(TextureImage::new(2, 2, vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 0, 0, 0, 0]).unwrap())
    }

    fn front_layer() -> TextureLayer {
        // On the +Z face of a unit cube centred at the origin, seen from +Z.
        TextureLayer {
            name: "Label".into(),
            image: checker(),
            origin: Vec3::new(-0.5, 0.5, 0.5),
            u_axis: Vec3::new(1.0, 0.0, 0.0),
            v_axis: Vec3::new(0.0, -1.0, 0.0),
            depth: 0.05,
            opacity: 1.0,
        }
    }

    #[test]
    fn a_layer_lands_only_on_the_face_it_looks_at() {
        let layer = front_layer();
        layer.validate().unwrap();
        assert_eq!(layer.normal(), Vec3::Z);
        // Top-left quarter of the front face is the red pixel.
        let red = layer.sample(Vec3::new(-0.25, 0.25, 0.5), Vec3::Z).unwrap();
        assert!(red[0] > 0.99 && red[1] < 0.01 && red[3] > 0.99);
        let blue = layer.sample(Vec3::new(-0.25, -0.25, 0.5), Vec3::Z).unwrap();
        assert!(blue[2] > 0.99 && blue[0] < 0.01);
        // Not on the back face, a side face, or outside the rectangle.
        assert!(layer.sample(Vec3::new(-0.25, 0.25, -0.5), Vec3::NEG_Z).is_none());
        assert!(layer.sample(Vec3::new(-0.25, 0.25, -0.5), Vec3::Z).is_none(), "too deep");
        assert!(layer.sample(Vec3::new(0.5, 0.25, 0.4), Vec3::X).is_none());
        assert!(layer.sample(Vec3::new(0.7, 0.25, 0.5), Vec3::Z).is_none());
        // The transparent pixel leaves the part's colour alone; the red one replaces it.
        let base = [0.2, 0.3, 0.4];
        assert_eq!(apply_layers(&[layer.clone()], base, Vec3::new(0.25, -0.25, 0.5), Vec3::Z), base);
        assert_eq!(apply_layers(&[layer.clone()], base, Vec3::new(-0.25, 0.25, 0.5), Vec3::Z), [1.0, 0.0, 0.0]);
        assert!((layer.texel_size() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn layers_survive_a_save_and_text_is_drawn() {
        let json = serde_json::to_string(&front_layer()).unwrap();
        assert!(json.contains("\"png:"));
        let back: TextureLayer = serde_json::from_str(&json).unwrap();
        assert_eq!(back.image, front_layer().image);

        let style = TextStyle { color: [1.0; 3], background: None, bold: true, padding: 0.1 };
        let title = render_text("Moby\nDick", 256, 128, &style).unwrap();
        let inked = title.rgba.chunks_exact(4).filter(|p| p[3] > 128).count();
        assert!(inked > 500 && inked < 256 * 128 / 2, "{inked} inked pixels");
        // Nothing touches the padding border.
        assert!((0..256).all(|x| title.rgba[(x * 4 + 3) as usize] == 0));
        let solid = TextStyle { color: [0.0; 3], background: Some([1.0; 3]), bold: false, padding: 0.1 };
        let label = render_text("EXIT", 128, 64, &solid).unwrap();
        assert!(label.rgba.chunks_exact(4).all(|p| p[3] == 255));
        assert!(label.rgba.chunks_exact(4).any(|p| p[0] < 40) && label.rgba[0] == 255);
        assert!(render_text("  ", 64, 64, &style).is_err());
        assert!(TextureImage::new(4096, 16, Vec::new()).is_err());
    }

    #[test]
    fn image_coordinates_follow_skewed_axes() {
        let mut layer = front_layer();
        layer.v_axis += layer.u_axis * 0.5;
        layer.validate().unwrap();
        let point = layer.origin + layer.u_axis * 0.75 + layer.v_axis * 0.75;
        assert_eq!(layer.uv(point, Vec3::Z), Some((0.75, 0.75)));
    }
}
