use crate::camera::OrbitCamera;
use crate::gizmo::GizmoState;
use crate::mesh::{EditElementMode, GpuVertex};
use crate::scene::{InteractionMode, Scene};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GlobalUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub clip_plane: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ObjectUniforms {
    pub model: [[f32; 4]; 4],
    pub normal_mat: [[f32; 4]; 4],
    pub params: [f32; 4],
    pub tex_params: [f32; 4],
    pub emission: [f32; 4],
    pub inspection: [f32; 4],
    /// x = number of image layers.
    pub layer_info: [f32; 4],
    /// Five vec4 per layer: origin+opacity, u+depth, v, normal, atlas rect (shader `ImageLayer`).
    pub layers: [[f32; 4]; 20],
}

/// A part's image layers on the GPU: one atlas texture and the uniform rows that place them.
struct LayerAtlas {
    /// The images it was built from (address, width, height), to notice replaced layers.
    key: Vec<(usize, u32, u32)>,
    view: wgpu::TextureView,
    /// Atlas rectangle of each layer: corner and size in 0..1, inset by half a texel.
    rects: Vec<[f32; 4]>,
}

fn layer_atlas_key(layers: &[crate::texture::TextureLayer]) -> Vec<(usize, u32, u32)> {
    layers
        .iter()
        .map(|l| (std::sync::Arc::as_ptr(&l.image) as usize, l.image.width, l.image.height))
        .collect()
}

fn layer_uniforms(layers: &[crate::texture::TextureLayer], rects: &[[f32; 4]]) -> ([f32; 4], [[f32; 4]; 20]) {
    let mut rows = [[0.0_f32; 4]; 20];
    let count = layers.len().min(crate::texture::MAX_LAYERS).min(rects.len());
    for (i, layer) in layers.iter().take(count).enumerate() {
        let (u, v, n) = (layer.u_axis, layer.v_axis, layer.normal());
        let (u, v) = (u / u.length_squared(), v / v.length_squared());
        rows[i * 5] = [layer.origin.x, layer.origin.y, layer.origin.z, layer.opacity.clamp(0.0, 1.0)];
        rows[i * 5 + 1] = [u.x, u.y, u.z, layer.depth];
        rows[i * 5 + 2] = [v.x, v.y, v.z, 0.0];
        rows[i * 5 + 3] = [n.x, n.y, n.z, 0.0];
        rows[i * 5 + 4] = rects[i];
    }
    ([count as f32, 0.0, 0.0, 0.0], rows)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ViewportShading {
    StudioLit,
    SculptClay,
    WireframeOnly,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
struct MeshCacheKey {
    mesh_version: u64,
    color_bits: [u32; 3],
    flat_shading: bool,
    is_selected: bool,
    is_edit_mode: bool,
    /// The full wireframe is only built while it is shown; it is costly on dense meshes.
    wire: bool,
}

struct CachedGpuMesh {
    key: MeshCacheKey,
    tri_buffer: Option<(wgpu::Buffer, u32)>,
    wire_buffer: Option<(wgpu::Buffer, u32)>,
    outline_buffer: Option<(wgpu::Buffer, u32)>,
}

/// Cache key: object id plus which comparison side it is drawn for.
type DrawKey = (u64, u8);

struct ObjectDraws {
    solid: Vec<(DrawKey, wgpu::BindGroup)>,
    ghost: Vec<(DrawKey, wgpu::BindGroup, f32)>,
    /// `true` draws the full wireframe; the selection outline is drawn either way.
    wire: Vec<(DrawKey, wgpu::BindGroup, bool)>,
    min: Vec3,
    max: Vec3,
}

impl Default for ObjectDraws {
    fn default() -> Self {
        Self {
            solid: Vec::new(),
            ghost: Vec::new(),
            wire: Vec::new(),
            min: Vec3::splat(f32::MAX),
            max: Vec3::splat(f32::MIN),
        }
    }
}

pub struct PreparedDrawCall {
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count: u32,
    pub bind_group: wgpu::BindGroup,
}

pub struct Renderer3D {
    pub triangle_pipeline: wgpu::RenderPipeline,
    pub ghost_pipeline: wgpu::RenderPipeline,
    pub line_depth_pipeline: wgpu::RenderPipeline,
    pub line_overlay_pipeline: wgpu::RenderPipeline,
    pub line_xray_pipeline: wgpu::RenderPipeline,
    background_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_vertex_buffer: wgpu::Buffer,
    /// Split position (0..1 of the viewport width) for before/after wipe comparisons.
    pub compare_split: Option<f32>,
    pub global_buffer: wgpu::Buffer,
    pub global_bind_group: wgpu::BindGroup,
    pub object_bind_group_layout: wgpu::BindGroupLayout,
    pub identity_line_bind_group: wgpu::BindGroup,
    pub depth_texture_view: wgpu::TextureView,
    msaa_view: wgpu::TextureView,
    surface_format: wgpu::TextureFormat,
    pub depth_size: (u32, u32),
    pub grid_vertex_buffer: wgpu::Buffer,
    pub grid_vertex_count: u32,
    pub ghost_unselected: bool,
    /// Draw selection outlines through occluding geometry.
    pub xray_selection: bool,
    pub show_grid: bool,
    pub show_wireframe_overlay: bool,
    pub shading: ViewportShading,
    pub viewport: Option<[f32; 4]>,
    mesh_cache: HashMap<DrawKey, CachedGpuMesh>,
    /// Image-layer atlases of textured parts.
    layer_atlases: HashMap<DrawKey, LayerAtlas>,
    blank_layers: wgpu::TextureView,
    layer_sampler: wgpu::Sampler,
    /// Hover-preview outline, rebuilt only when the hovered face set or its mesh changes.
    hover_lines: Option<(HoverKey, wgpu::Buffer, u32)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct HoverKey {
    object_id: u64,
    mesh_version: u64,
    face_count: usize,
    face_hash: u64,
    model_bits: [u32; 16],
}

impl HoverKey {
    fn new(obj: &crate::scene::SceneObject, faces: &std::collections::HashSet<usize>) -> Self {
        // Order-independent hash of the face set.
        let face_hash = faces.iter().fold(0u64, |acc, &f| {
            let mut z = (f as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            acc.wrapping_add(z ^ (z >> 31))
        });
        Self {
            object_id: obj.id,
            mesh_version: obj.mesh.version,
            face_count: faces.len(),
            face_hash,
            model_bits: obj.transform.matrix().to_cols_array().map(f32::to_bits),
        }
    }
}

impl Renderer3D {
    pub fn invalidate_cache(&mut self) {
        self.mesh_cache.clear();
        self.layer_atlases.clear();
        self.hover_lines = None;
    }

    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    /// Anti-aliasing samples for the 3D pass (resolved before the UI is drawn).
    const SAMPLES: u32 = 4;

    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("3D Studio Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let global_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Global Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let object_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Object Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let global_uniforms = GlobalUniforms {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 3.0, 6.0, 1.0],
            clip_plane: [0.0, 0.0, 0.0, 1.0],
        };

        let global_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Uniform Buffer"),
            contents: bytemuck::cast_slice(&[global_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Global Bind Group"),
            layout: &global_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_buffer.as_entire_binding(),
            }],
        });

        let identity_u = ObjectUniforms {
            model: Mat4::IDENTITY.to_cols_array_2d(),
            normal_mat: Mat4::IDENTITY.to_cols_array_2d(),
            params: [0.5, 0.0, 2.0, 0.0],
            tex_params: [0.3, 0.3, 0.3, 1.0],
            emission: [0.0, 0.0, 0.0, 0.0],
            inspection: [1.0, 0.0, 0.0, 0.0],
            layer_info: [0.0; 4],
            layers: [[0.0; 4]; 20],
        };
        // Parts without image layers bind this one-pixel texture.
        let blank_layers = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Blank Layer Atlas"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let layer_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Layer Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let identity_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Identity Line Object Uniform Buffer"),
            contents: bytemuck::cast_slice(&[identity_u]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let identity_line_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Identity Line Bind Group"),
            layout: &object_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: identity_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&blank_layers) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&layer_sampler) },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("3D Pipeline Layout"),
            bind_group_layouts: &[&global_bind_group_layout, &object_bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }];

        let triangle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Triangle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 1.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: Self::SAMPLES,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        let ghost_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Inspection Ghost Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 1.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: Self::SAMPLES,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        let line_depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Depth Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: Self::SAMPLES,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        let line_overlay_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Line Overlay Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                count: Self::SAMPLES,
                ..Default::default()
            },
                multiview: None,
                cache: None,
            });

        // Same lines drawn through geometry at reduced strength (blend constant).
        let line_xray_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Line X-ray Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Constant,
                                dst_factor: wgpu::BlendFactor::OneMinusConstant,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::OVER,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                count: Self::SAMPLES,
                ..Default::default()
            },
                multiview: None,
                cache: None,
            });

        let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Backdrop Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_backdrop"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_backdrop"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: Self::SAMPLES,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Contact Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_shadow"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: Self::SAMPLES,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });
        let quad: Vec<GpuVertex> = [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
        .iter()
        .map(|[x, z]| GpuVertex {
            position: [*x, 0.0, *z],
            normal: [0.0, 1.0, 0.0],
            color: [0.0, 0.0, 0.0],
        })
        .collect();
        let shadow_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Contact Shadow Quad"),
            contents: bytemuck::cast_slice(&quad),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let w = width.max(1);
        let h = height.max(1);
        let depth_texture_view = Self::create_depth_view(device, w, h);
        let msaa_view = Self::create_msaa_view(device, surface_format, w, h);
        let grid_verts = Self::build_grid_vertices(12, 1.0);
        let grid_vertex_count = grid_verts.len() as u32;
        let grid_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&grid_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            triangle_pipeline,
            ghost_pipeline,
            line_depth_pipeline,
            line_overlay_pipeline,
            line_xray_pipeline,
            background_pipeline,
            shadow_pipeline,
            shadow_vertex_buffer,
            compare_split: None,
            global_buffer,
            global_bind_group,
            object_bind_group_layout,
            identity_line_bind_group,
            depth_texture_view,
            msaa_view,
            surface_format,
            depth_size: (w, h),
            grid_vertex_buffer,
            grid_vertex_count,
            show_grid: true,
            ghost_unselected: false,
            xray_selection: true,
            show_wireframe_overlay: false,
            shading: ViewportShading::StudioLit,
            viewport: None,
            mesh_cache: HashMap::new(),
            layer_atlases: HashMap::new(),
            blank_layers,
            layer_sampler,
            hover_lines: None,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.depth_size != (w, h) {
            self.depth_texture_view = Self::create_depth_view(device, w, h);
            self.msaa_view = Self::create_msaa_view(device, self.surface_format, w, h);
            self.depth_size = (w, h);
        }
    }

    /// Multisampled colour target the 3D pass draws into before resolving to the output.
    fn create_msaa_view(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Colour Buffer"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: Self::SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Buffer"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: Self::SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn build_grid_vertices(half_extent: i32, step: f32) -> Vec<GpuVertex> {
        let mut verts = Vec::new();
        let max_coord = (half_extent as f32) * step;

        for i in -half_extent..=half_extent {
            let coord = (i as f32) * step;
            let z_col = if i == 0 {
                [0.75, 0.25, 0.25]
            } else if i % 5 == 0 {
                [0.12, 0.14, 0.17]
            } else {
                [0.065, 0.08, 0.10]
            };
            verts.push(GpuVertex {
                position: [-max_coord, 0.0, coord],
                normal: [0.0, 1.0, 0.0],
                color: z_col,
            });
            verts.push(GpuVertex {
                position: [max_coord, 0.0, coord],
                normal: [0.0, 1.0, 0.0],
                color: z_col,
            });

            let x_col = if i == 0 {
                [0.25, 0.48, 0.85]
            } else if i % 5 == 0 {
                [0.12, 0.14, 0.17]
            } else {
                [0.065, 0.08, 0.10]
            };
            verts.push(GpuVertex {
                position: [coord, 0.0, -max_coord],
                normal: [0.0, 1.0, 0.0],
                color: x_col,
            });
            verts.push(GpuVertex {
                position: [coord, 0.0, max_coord],
                normal: [0.0, 1.0, 0.0],
                color: x_col,
            });
        }
        verts
    }

    #[allow(clippy::too_many_arguments)]
    fn create_object_bind_group(
        &self,
        device: &wgpu::Device,
        model: Mat4,
        roughness: f32,
        metallic: f32,
        shading_mode: f32,
        pattern_id: f32,
        secondary_color: [f32; 3],
        texture_scale: f32,
        emission_color: [f32; 3],
        emission_strength: f32,
        opacity: f32,
        glow: f32,
        layers: Option<(&wgpu::TextureView, [f32; 4], [[f32; 4]; 20])>,
    ) -> wgpu::BindGroup {
        let normal_mat = model.inverse().transpose();
        let (atlas, layer_info, layer_rows) = match layers {
            Some((view, info, rows)) => (view, info, rows),
            None => (&self.blank_layers, [0.0; 4], [[0.0; 4]; 20]),
        };
        let u = ObjectUniforms {
            layer_info,
            layers: layer_rows,
            model: model.to_cols_array_2d(),
            normal_mat: normal_mat.to_cols_array_2d(),
            params: [roughness, metallic, shading_mode, pattern_id],
            inspection: [opacity, glow, 0.0, 0.0],
            tex_params: [
                secondary_color[0],
                secondary_color[1],
                secondary_color[2],
                texture_scale,
            ],
            emission: [
                emission_color[0],
                emission_color[1],
                emission_color[2],
                emission_strength,
            ],
        };
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Object Uniform Buffer"),
            contents: bytemuck::cast_slice(&[u]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Object Bind Group"),
            layout: &self.object_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(atlas) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.layer_sampler) },
            ],
        })
    }

    /// Makes sure the part's image layers are on the GPU as one atlas (a 2x2 grid of cells
    /// with a gutter, so neighbours never bleed into each other).
    fn refresh_layer_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: DrawKey,
        layers: &[crate::texture::TextureLayer],
    ) {
        use wgpu::util::DeviceExt;
        let layers = &layers[..layers.len().min(crate::texture::MAX_LAYERS)];
        let atlas_key = layer_atlas_key(layers);
        if self.layer_atlases.get(&key).is_some_and(|a| a.key == atlas_key) {
            return;
        }
        const GUTTER: u32 = 2;
        let cell_w = layers.iter().map(|l| l.image.width).max().unwrap_or(1) + GUTTER;
        let cell_h = layers.iter().map(|l| l.image.height).max().unwrap_or(1) + GUTTER;
        let columns = if layers.len() > 1 { 2 } else { 1 };
        let rows = (layers.len() as u32).div_ceil(columns).max(1);
        let (width, height) = (cell_w * columns, cell_h * rows);
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        let mut rects = Vec::new();
        for (i, layer) in layers.iter().enumerate() {
            let image = &layer.image;
            let (x0, y0) = ((i as u32 % columns) * cell_w + GUTTER / 2, (i as u32 / columns) * cell_h + GUTTER / 2);
            for row in 0..image.height {
                let from = (row * image.width * 4) as usize;
                let to = (((y0 + row) * width + x0) * 4) as usize;
                let len = (image.width * 4) as usize;
                pixels[to..to + len].copy_from_slice(&image.rgba[from..from + len]);
            }
            rects.push([
                (x0 as f32 + 0.5) / width as f32,
                (y0 as f32 + 0.5) / height as f32,
                (image.width as f32 - 1.0).max(0.0) / width as f32,
                (image.height as f32 - 1.0).max(0.0) / height as f32,
            ]);
        }
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Layer Atlas"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &pixels,
        );
        self.layer_atlases.insert(
            key,
            LayerAtlas {
                key: atlas_key,
                view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
                rects,
            },
        );
    }

    fn create_line_bind_group(&self, device: &wgpu::Device, model: Mat4) -> wgpu::BindGroup {
        self.create_object_bind_group(
            device,
            model,
            0.5,
            0.0,
            2.0,
            0.0,
            [0.3, 0.3, 0.3],
            1.0,
            [0.0, 0.0, 0.0],
            0.0,
            1.0,
            0.0,
            None,
        )
    }

    /// Rebuild the cached GPU buffers of every visible part whose mesh, selection or look changed.
    ///
    /// Vertex data is built on worker threads: one selection change over a dense scene (a 3D box
    /// drag, a box select) can dirty dozens of parts in a single frame.
    fn refresh_mesh_cache(
        &mut self,
        device: &wgpu::Device,
        scene: &Scene,
        variant: u8,
        is_edit_mode: bool,
        want_wire: bool,
    ) {
        let mut stale = Vec::new();
        for (i, obj) in scene.objects.iter().enumerate() {
            if !obj.visible {
                continue;
            }
            let is_selected = scene.selected_object == Some(i);
            let key = MeshCacheKey {
                mesh_version: obj.mesh.version,
                color_bits: [
                    obj.color[0].to_bits(),
                    obj.color[1].to_bits(),
                    obj.color[2].to_bits(),
                ],
                flat_shading: obj.flat_shading,
                is_selected,
                is_edit_mode: !obj.mesh.selected_faces.is_empty() || (is_selected && is_edit_mode),
                wire: want_wire,
            };
            let cache_key = (obj.id, variant);
            if self
                .mesh_cache
                .get(&cache_key)
                .is_none_or(|cached| cached.key != key)
            {
                stale.push((cache_key, key, obj));
            }
        }
        if stale.is_empty() {
            return;
        }

        type Built = [Vec<GpuVertex>; 3];
        let build = |&(_, key, obj): &(DrawKey, MeshCacheKey, &crate::scene::SceneObject)| -> Built {
            let tris = obj.mesh.build_gpu_triangles(obj.color, obj.flat_shading);
            let wire_col = if key.is_selected {
                [0.40, 0.89, 0.76]
            } else {
                [0.32, 0.35, 0.40]
            };
            let lines = if key.wire {
                obj.mesh
                    .build_wireframe_lines(Mat4::IDENTITY, wire_col, key.is_edit_mode)
            } else {
                Vec::new()
            };
            let outline = if obj.mesh.selected_faces.is_empty() {
                Vec::new()
            } else {
                obj.mesh.build_selection_outline([0.55, 1.0, 0.88])
            };
            [tris, lines, outline]
        };
        let make = |label: &str, verts: &[GpuVertex]| {
            (!verts.is_empty()).then(|| {
                (
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: bytemuck::cast_slice(verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    verts.len() as u32,
                )
            })
        };
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());

        // Batches bound how much vertex data is held in memory before it is uploaded.
        const BATCH_FACES: usize = 400_000;
        let mut rest = &stale[..];
        while !rest.is_empty() {
            let mut faces = 0;
            let len = rest
                .iter()
                .take_while(|(_, _, obj)| {
                    let fits = faces < BATCH_FACES;
                    faces += obj.mesh.faces.len();
                    fits
                })
                .count();
            let (batch, tail) = rest.split_at(len);
            rest = tail;
            let workers = threads.min(batch.len());
            let built: Vec<Built> = if workers <= 1 {
                batch.iter().map(&build).collect()
            } else {
                // Workers take the next part off a shared counter, so one dense part does not
                // hold up a whole chunk.
                let next = std::sync::atomic::AtomicUsize::new(0);
                let mut done: Vec<(usize, Built)> = std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..workers)
                        .map(|_| {
                            scope.spawn(|| {
                                let mut out = Vec::new();
                                loop {
                                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    let Some(item) = batch.get(i) else { break };
                                    out.push((i, build(item)));
                                }
                                out
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                        .collect()
                });
                done.sort_unstable_by_key(|(i, _)| *i);
                done.into_iter().map(|(_, b)| b).collect()
            };
            for (&(cache_key, key, _), [tris, lines, outline]) in batch.iter().zip(&built) {
                self.mesh_cache.insert(
                    cache_key,
                    CachedGpuMesh {
                        key,
                        tri_buffer: make("Cached Mesh Tri VB", tris),
                        wire_buffer: make("Cached Mesh Wire VB", lines),
                        outline_buffer: make("Cached Selection Outline VB", outline),
                    },
                );
            }
        }
    }

    /// Upload/refresh cached GPU buffers and build per-object draw lists for `scene`.
    fn prepare_objects(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        variant: u8,
        eye: Vec3,
    ) -> ObjectDraws {
        let is_edit_mode = matches!(scene.mode, InteractionMode::EditMesh(_));
        let any_selection = scene.has_any_selection();
        let mut draws = ObjectDraws::default();
        let default_mode_val = match self.shading {
            ViewportShading::StudioLit => {
                if matches!(scene.mode, InteractionMode::Sculpt) {
                    1.0
                } else {
                    0.0
                }
            }
            ViewportShading::SculptClay => 1.0,
            ViewportShading::WireframeOnly => 2.0,
        };
        let want_wire =
            self.shading == ViewportShading::WireframeOnly || self.show_wireframe_overlay;
        self.refresh_mesh_cache(device, scene, variant, is_edit_mode, want_wire);
        for (i, obj) in scene.objects.iter().enumerate() {
            if !obj.visible {
                continue;
            }
            let is_selected = scene.selected_object == Some(i);
            let has_face_sel =
                !obj.mesh.selected_faces.is_empty() || (is_selected && is_edit_mode);
            let cache_key = (obj.id, variant);

            let model = obj.transform.matrix();
            let ghost = self.ghost_unselected && any_selection && !is_selected && !has_face_sel;
            if self.shading != ViewportShading::WireframeOnly {
                // Whole-part selections glow; face selections are tinted and outlined.
                let glow = if is_selected && obj.mesh.selected_faces.is_empty() {
                    1.0
                } else {
                    0.0
                };
                if !obj.textures.is_empty() {
                    self.refresh_layer_atlas(device, queue, cache_key, &obj.textures);
                }
                let layers = self.layer_atlases.get(&cache_key).filter(|_| !obj.textures.is_empty()).map(|atlas| {
                    let (info, rows) = layer_uniforms(&obj.textures, &atlas.rects);
                    (&atlas.view, info, rows)
                });
                let bg = self.create_object_bind_group(
                    device,
                    model,
                    obj.roughness,
                    obj.metallic,
                    default_mode_val,
                    obj.pattern.shader_id(),
                    obj.secondary_color,
                    obj.texture_scale,
                    obj.emission_color,
                    obj.emission_strength,
                    if ghost { 0.16 } else { 1.0 },
                    glow,
                    layers,
                );
                if ghost {
                    draws.ghost.push((
                        cache_key,
                        bg,
                        (obj.transform.translation - eye).length_squared(),
                    ));
                } else {
                    draws.solid.push((cache_key, bg));
                }
            }
            if want_wire || !obj.mesh.selected_faces.is_empty() {
                draws.wire.push((
                    cache_key,
                    self.create_line_bind_group(device, model),
                    want_wire,
                ));
            }
            let (min, max) = obj.mesh.bounds();
            for c in [
                Vec3::new(min.x, min.y, min.z),
                Vec3::new(max.x, min.y, min.z),
                Vec3::new(min.x, max.y, min.z),
                Vec3::new(min.x, min.y, max.z),
                Vec3::new(max.x, max.y, min.z),
                Vec3::new(max.x, min.y, max.z),
                Vec3::new(min.x, max.y, max.z),
                Vec3::new(max.x, max.y, max.z),
            ] {
                let p = model.transform_point3(c);
                draws.min = draws.min.min(p);
                draws.max = draws.max.max(p);
            }
        }
        draws.ghost.sort_by(|a, b| b.2.total_cmp(&a.2));
        draws
    }

    fn draw_objects<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, draws: &'a ObjectDraws) {
        if !draws.solid.is_empty() {
            rpass.set_pipeline(&self.triangle_pipeline);
            rpass.set_bind_group(0, &self.global_bind_group, &[]);
            for (key, bg) in &draws.solid {
                if let Some((vb, count)) = self.mesh_cache.get(key).and_then(|m| m.tri_buffer.as_ref()) {
                    rpass.set_bind_group(1, bg, &[]);
                    rpass.set_vertex_buffer(0, vb.slice(..));
                    rpass.draw(0..*count, 0..1);
                }
            }
        }
        if !draws.ghost.is_empty() {
            rpass.set_pipeline(&self.ghost_pipeline);
            rpass.set_bind_group(0, &self.global_bind_group, &[]);
            for (key, bg, _) in &draws.ghost {
                if let Some((vb, count)) = self.mesh_cache.get(key).and_then(|m| m.tri_buffer.as_ref()) {
                    rpass.set_bind_group(1, bg, &[]);
                    rpass.set_vertex_buffer(0, vb.slice(..));
                    rpass.draw(0..*count, 0..1);
                }
            }
        }
        if !draws.wire.is_empty() {
            rpass.set_pipeline(&self.line_depth_pipeline);
            rpass.set_bind_group(0, &self.global_bind_group, &[]);
            for (key, bg, full) in &draws.wire {
                let Some(cached) = self.mesh_cache.get(key) else {
                    continue;
                };
                rpass.set_bind_group(1, bg, &[]);
                if *full {
                    if let Some((vb, count)) = &cached.wire_buffer {
                        rpass.set_vertex_buffer(0, vb.slice(..));
                        rpass.draw(0..*count, 0..1);
                    }
                }
                if let Some((vb, count)) = &cached.outline_buffer {
                    rpass.set_vertex_buffer(0, vb.slice(..));
                    rpass.draw(0..*count, 0..1);
                }
            }
            // Selection outlines again without depth testing, so a selected region stays visible
            // behind other geometry.
            if self.xray_selection {
                rpass.set_pipeline(&self.line_xray_pipeline);
                rpass.set_blend_constant(wgpu::Color { r: 0.4, g: 0.4, b: 0.4, a: 1.0 });
                rpass.set_bind_group(0, &self.global_bind_group, &[]);
                for (key, bg, _) in &draws.wire {
                    if let Some((vb, count)) =
                        self.mesh_cache.get(key).and_then(|c| c.outline_buffer.as_ref())
                    {
                        rpass.set_bind_group(1, bg, &[]);
                        rpass.set_vertex_buffer(0, vb.slice(..));
                        rpass.draw(0..*count, 0..1);
                    }
                }
            }
        }
    }

    /// Soft contact shadow under the scene's footprint.
    fn shadow_bind_group(&self, device: &wgpu::Device, min: Vec3, max: Vec3) -> Option<wgpu::BindGroup> {
        if !(min.x <= max.x) || !min.is_finite() || !max.is_finite() {
            return None;
        }
        let center = (min + max) * 0.5;
        let half = ((max - min) * 0.5).max(Vec3::splat(0.05));
        let radius = half.x.max(half.z) * 1.15;
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(radius, 1.0, radius),
            glam::Quat::IDENTITY,
            Vec3::new(center.x, min.y - 0.002, center.z),
        );
        Some(self.create_object_bind_group(
            device,
            model,
            0.5,
            0.0,
            0.0,
            0.0,
            [0.0; 3],
            1.0,
            [0.0; 3],
            0.0,
            1.0,
            0.0,
            None,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        camera: &OrbitCamera,
        scene: &Scene,
        gizmo: &GizmoState,
        brush_cursor: Option<(Vec3, Vec3, f32)>,
        hover_preview: Option<(u64, &std::collections::HashSet<usize>)>,
    ) {
        let eye = camera.eye();
        let global_u = GlobalUniforms {
            view_proj: camera.view_proj_matrix().to_cols_array_2d(),
            camera_pos: [eye.x, eye.y, eye.z, 1.0],
            clip_plane: scene.clip_plane.unwrap_or([0.0, 0.0, 0.0, 1.0]),
        };
        queue.write_buffer(&self.global_buffer, 0, bytemuck::cast_slice(&[global_u]));

        // Split comparison: "before" on the left of the divider, "after" on the right.
        let split = scene
            .preview_revision
            .and(self.compare_split)
            .map(|s| s.clamp(0.0, 1.0));
        let mut passes: Vec<(ObjectDraws, Option<[u32; 4]>)> = Vec::new();
        let [vx, vy, vw, vh] = self.viewport.unwrap_or([
            0.0,
            0.0,
            self.depth_size.0 as f32,
            self.depth_size.1 as f32,
        ]);
        let scissor = |from: f32, to: f32, size: (u32, u32)| -> [u32; 4] {
            let x0 = (vx + vw * from).round().clamp(0.0, size.0 as f32) as u32;
            let x1 = (vx + vw * to).round().clamp(0.0, size.0 as f32) as u32;
            let y0 = vy.round().clamp(0.0, size.1 as f32) as u32;
            let y1 = (vy + vh).round().clamp(0.0, size.1 as f32) as u32;
            [x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)]
        };
        let display = scene.display_scene();
        if let (Some(s), Some((rid, _))) = (split, scene.preview_revision) {
            let before = scene.revision_view(rid, true);
            let after = scene.revision_view(rid, false);
            if let (Some(before), Some(after)) = (before, after) {
                let left = self.prepare_objects(device, queue, &before, 1, eye);
                passes.push((left, Some(scissor(0.0, s, self.depth_size))));
                let right = self.prepare_objects(device, queue, &after, 0, eye);
                passes.push((right, Some(scissor(s, 1.0, self.depth_size))));
            }
        }
        if passes.is_empty() {
            let draws = self.prepare_objects(device, queue, display.as_ref(), 0, eye);
            passes.push((draws, None));
        }
        let scene = display.as_ref();
        let mut bounds_min = Vec3::splat(f32::MAX);
        let mut bounds_max = Vec3::splat(f32::MIN);
        for (d, _) in &passes {
            bounds_min = bounds_min.min(d.min);
            bounds_max = bounds_max.max(d.max);
        }
        let shadow = self.show_grid.then(|| self.shadow_bind_group(device, bounds_min, bounds_max)).flatten();

        let mut overlay_line_draws: Vec<PreparedDrawCall> = Vec::new();
        let is_edit_mode = matches!(scene.mode, InteractionMode::EditMesh(_));
        for (i, obj) in scene.objects.iter().enumerate() {
            if !obj.visible
                || scene.selected_object != Some(i)
                || !matches!(scene.mode, InteractionMode::EditMesh(EditElementMode::Vertex))
            {
                continue;
            }
            let _ = is_edit_mode;
            let model = obj.transform.matrix();
            let mut v_lines = Vec::new();
            let r = 0.035;
            for (vi, &p) in obj.mesh.positions.iter().enumerate() {
                let wp = model.transform_point3(p);
                let sel = obj.mesh.selected_vertices.contains(&vi);
                let col = if sel { [1.0, 0.72, 0.1] } else { [0.45, 0.52, 0.62] };
                let s = if sel { r * 1.4 } else { r * 0.75 };
                for axis in [Vec3::X * s, Vec3::Y * s, Vec3::Z * s] {
                    for q in [wp - axis, wp + axis] {
                        v_lines.push(GpuVertex {
                            position: q.to_array(),
                            normal: [0.0, 1.0, 0.0],
                            color: col,
                        });
                    }
                }
            }
            if !v_lines.is_empty() {
                let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Cross VB"),
                    contents: bytemuck::cast_slice(&v_lines),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                overlay_line_draws.push(PreparedDrawCall {
                    vertex_buffer: vb,
                    vertex_count: v_lines.len() as u32,
                    bind_group: self.create_line_bind_group(device, Mat4::IDENTITY),
                });
            }
        }

        // Live Hover Preview Wireframe (Smart Sub-Part / Smart Surface highlight before clicking)
        let mut draw_hover = false;
        if let Some((hover_obj_id, hover_faces)) = hover_preview {
            if !hover_faces.is_empty() {
                if let Some(obj) = scene
                    .objects
                    .iter()
                    .find(|o| o.id == hover_obj_id && o.visible)
                {
                    let key = HoverKey::new(obj, hover_faces);
                    if self.hover_lines.as_ref().is_none_or(|(k, _, _)| *k != key) {
                        let preview_lines = obj.mesh.build_face_subset_lines(
                            hover_faces,
                            obj.transform.matrix(),
                            [0.15, 0.94, 1.0],
                        );
                        self.hover_lines = (!preview_lines.is_empty()).then(|| {
                            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Hover Preview Wireframe VB"),
                                contents: bytemuck::cast_slice(&preview_lines),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                            (key, vb, preview_lines.len() as u32)
                        });
                    }
                    draw_hover = self.hover_lines.is_some();
                }
            }
        }

        // 3D Spatial Comment Pin & Region Markers (for AI-driven review workflow)
        let mut pin_lines = Vec::new();
        for c in &scene.comments {
            // Only the focused comment draws 3D markers; the UI badges mark the rest.
            if scene.active_comment_id != Some(c.id)
                || c.resolved
                || c.is_general
                || c.object_id == 0
                || scene.comment_needs_reattachment(c)
            {
                continue;
            }
            let col = [0.18, 0.95, 1.0];
            let r = c.selection_radius.clamp(0.15, 2.5);
            if !c.targets.is_empty() {
                for &(anchor, normal) in &scene.comment_geometry(c).pins {
                    pin_lines.extend(Self::build_comment_pin_lines(anchor, normal, r, col));
                }
            } else {
                pin_lines.extend(Self::build_comment_pin_lines(
                    c.world_anchor,
                    c.surface_normal,
                    r,
                    col,
                ));
            }
        }
        if !pin_lines.is_empty() {
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Comment Pins VB"),
                contents: bytemuck::cast_slice(&pin_lines),
                usage: wgpu::BufferUsages::VERTEX,
            });
            overlay_line_draws.push(PreparedDrawCall {
                vertex_buffer: vb,
                vertex_count: pin_lines.len() as u32,
                bind_group: self.create_line_bind_group(device, Mat4::IDENTITY),
            });
        }
        let _ = gizmo;

        // Sculpt / Voxel 3D Brush Ring Cursor
        if let Some((hit_pos, hit_normal, radius)) = brush_cursor {
            let ring_lines = Self::build_brush_cursor_lines(hit_pos, hit_normal, radius);
            if !ring_lines.is_empty() {
                let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Brush Cursor VB"),
                    contents: bytemuck::cast_slice(&ring_lines),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                overlay_line_draws.push(PreparedDrawCall {
                    vertex_buffer: vb,
                    vertex_count: ring_lines.len() as u32,
                    bind_group: self.create_line_bind_group(device, Mat4::IDENTITY),
                });
            }
        }

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("3D Viewport Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.msaa_view,
                resolve_target: Some(color_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.020,
                        g: 0.026,
                        b: 0.038,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if let Some([x, y, w, h]) = self.viewport {
            rpass.set_viewport(x, y, w.max(1.0), h.max(1.0), 0.0, 1.0);
        }
        // 0. Studio backdrop gradient.
        rpass.set_pipeline(&self.background_pipeline);
        rpass.set_bind_group(0, &self.global_bind_group, &[]);
        rpass.set_bind_group(1, &self.identity_line_bind_group, &[]);
        rpass.draw(0..3, 0..1);

        // 1. Objects (one pass per comparison side).
        for (draws, rect) in &passes {
            if let Some([x, y, w, h]) = rect {
                if *w == 0 || *h == 0 {
                    continue;
                }
                rpass.set_scissor_rect(*x, *y, *w, *h);
            }
            self.draw_objects(&mut rpass, draws);
        }
        if passes.iter().any(|(_, r)| r.is_some()) {
            rpass.set_scissor_rect(0, 0, self.depth_size.0, self.depth_size.1);
        }

        // 2. Contact shadow, grid and hover preview.
        if let Some(bg) = &shadow {
            rpass.set_pipeline(&self.shadow_pipeline);
            rpass.set_bind_group(0, &self.global_bind_group, &[]);
            rpass.set_bind_group(1, bg, &[]);
            rpass.set_vertex_buffer(0, self.shadow_vertex_buffer.slice(..));
            rpass.draw(0..6, 0..1);
        }
        if (self.show_grid && self.grid_vertex_count > 0) || draw_hover {
            rpass.set_pipeline(&self.line_depth_pipeline);
            rpass.set_bind_group(0, &self.global_bind_group, &[]);
            rpass.set_bind_group(1, &self.identity_line_bind_group, &[]);
            if self.show_grid && self.grid_vertex_count > 0 {
                rpass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
                rpass.draw(0..self.grid_vertex_count, 0..1);
            }
            if draw_hover {
                if let Some((_, vb, count)) = &self.hover_lines {
                    rpass.set_vertex_buffer(0, vb.slice(..));
                    rpass.draw(0..*count, 0..1);
                }
            }
        }

        // 3. Overlay lines (pins, brush cursor).
        if !overlay_line_draws.is_empty() {
            rpass.set_pipeline(&self.line_overlay_pipeline);
            rpass.set_bind_group(0, &self.global_bind_group, &[]);
            for call in &overlay_line_draws {
                rpass.set_bind_group(1, &call.bind_group, &[]);
                rpass.set_vertex_buffer(0, call.vertex_buffer.slice(..));
                rpass.draw(0..call.vertex_count, 0..1);
            }
        }
    }

    fn build_brush_cursor_lines(center: Vec3, normal: Vec3, radius: f32) -> Vec<GpuVertex> {
        let n = if normal.length_squared() > 1e-6 {
            normal.normalize()
        } else {
            Vec3::Y
        };
        let up = if n.y.abs() < 0.95 { Vec3::Y } else { Vec3::X };
        let t1 = n.cross(up).normalize_or_zero();
        let t2 = n.cross(t1).normalize_or_zero();

        let segs = 40;
        let mut lines = Vec::with_capacity(segs * 4 + 2);
        let col_outer = [1.0, 0.48, 0.22];
        let col_inner = [1.0, 0.78, 0.35];

        for i in 0..segs {
            let a0 = 2.0 * std::f32::consts::PI * (i as f32) / (segs as f32);
            let a1 = 2.0 * std::f32::consts::PI * ((i + 1) as f32) / (segs as f32);
            let dir0 = t1 * a0.cos() + t2 * a0.sin();
            let dir1 = t1 * a1.cos() + t2 * a1.sin();

            lines.push(GpuVertex {
                position: (center + dir0 * radius).to_array(),
                normal: [0.0, 1.0, 0.0],
                color: col_outer,
            });
            lines.push(GpuVertex {
                position: (center + dir1 * radius).to_array(),
                normal: [0.0, 1.0, 0.0],
                color: col_outer,
            });

            lines.push(GpuVertex {
                position: (center + dir0 * (radius * 0.5)).to_array(),
                normal: [0.0, 1.0, 0.0],
                color: col_inner,
            });
            lines.push(GpuVertex {
                position: (center + dir1 * (radius * 0.5)).to_array(),
                normal: [0.0, 1.0, 0.0],
                color: col_inner,
            });
        }

        lines.push(GpuVertex {
            position: center.to_array(),
            normal: [0.0, 1.0, 0.0],
            color: col_inner,
        });
        lines.push(GpuVertex {
            position: (center + n * (radius * 0.45)).to_array(),
            normal: [0.0, 1.0, 0.0],
            color: col_inner,
        });

        lines
    }

    fn build_comment_pin_lines(
        anchor: Vec3,
        normal: Vec3,
        radius: f32,
        color: [f32; 3],
    ) -> Vec<GpuVertex> {
        let n = if normal.length_squared() > 1e-6 {
            normal.normalize()
        } else {
            Vec3::Y
        };
        let up = if n.y.abs() < 0.95 { Vec3::Y } else { Vec3::X };
        let t1 = n.cross(up).normalize_or_zero();
        let t2 = n.cross(t1).normalize_or_zero();

        let segs = 24;
        let mut lines = Vec::with_capacity(segs * 2 + 18);

        // Surface ring around the commented section
        for i in 0..segs {
            let a0 = 2.0 * std::f32::consts::PI * (i as f32) / (segs as f32);
            let a1 = 2.0 * std::f32::consts::PI * ((i + 1) as f32) / (segs as f32);
            let p0 = anchor + (t1 * a0.cos() + t2 * a0.sin()) * radius;
            let p1 = anchor + (t1 * a1.cos() + t2 * a1.sin()) * radius;
            lines.push(GpuVertex {
                position: p0.to_array(),
                normal: [0.0, 1.0, 0.0],
                color,
            });
            lines.push(GpuVertex {
                position: p1.to_array(),
                normal: [0.0, 1.0, 0.0],
                color,
            });
        }

        // Pin stem rising out of the surface + 3D diamond beacon at tip
        let pin_dir = (n * 0.65 + Vec3::Y * 0.75).normalize_or_zero();
        let stem_len = (radius * 1.15).clamp(0.45, 1.4);
        let tip = anchor + pin_dir * stem_len;
        lines.push(GpuVertex {
            position: anchor.to_array(),
            normal: [0.0, 1.0, 0.0],
            color,
        });
        lines.push(GpuVertex {
            position: tip.to_array(),
            normal: [0.0, 1.0, 0.0],
            color,
        });

        let d = 0.12;
        let diamond_pts = [
            tip + Vec3::Y * d,
            tip - Vec3::Y * d,
            tip + Vec3::X * d,
            tip - Vec3::X * d,
            tip + Vec3::Z * d,
            tip - Vec3::Z * d,
        ];
        for &side_pt in &diamond_pts[2..] {
            for &pole in &diamond_pts[0..2] {
                lines.push(GpuVertex {
                    position: pole.to_array(),
                    normal: [0.0, 1.0, 0.0],
                    color,
                });
                lines.push(GpuVertex {
                    position: side_pt.to_array(),
                    normal: [0.0, 1.0, 0.0],
                    color,
                });
            }
        }

        lines
    }

    /// Renders the current 3D scene to an offscreen texture and saves it as a PNG file.
    pub fn render_to_png_file(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &OrbitCamera,
        scene: &Scene,
        gizmo: &GizmoState,
        width: u32,
        height: u32,
        path: &std::path::Path,
        studio: Option<usize>,
    ) -> Result<(), String> {
        let w = width.clamp(64, 4096);
        let h = height.clamp(64, 4096);

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut offscreen_renderer = Self::new(device, format, w, h);
        offscreen_renderer.shading = self.shading;
        offscreen_renderer.ghost_unselected = self.ghost_unselected;
        offscreen_renderer.xray_selection = self.xray_selection;
        offscreen_renderer.show_wireframe_overlay = self.show_wireframe_overlay;
        offscreen_renderer.show_grid = self.show_grid;
        offscreen_renderer.compare_split = self.compare_split;

        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen Screenshot Texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut cam = camera.clone();
        cam.aspect = (w as f32) / (h as f32);

        let mut ui_scene = scene.clone();
        let mut ui_renderer = egui_wgpu::Renderer::new(device, format, None, 1, false);
        let mut ui_output = None;
        let ctx = egui::Context::default();
        if let Some(tab) = studio {
            crate::ui::apply_theme(&ctx);
            let mut state = crate::ui::UiState::default();
            state.review_tab = tab;
            state.compare_split = self.compare_split;
            let mut history = crate::history::UndoHistory::default();
            let mut ui_gizmo = GizmoState::default();
            // Step time generously so egui's area fade-in animations finish before capture.
            for frame in 0..4 {
                let input = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(w as f32, h as f32),
                    )),
                    time: Some(frame as f64 * 0.5),
                    ..Default::default()
                };
                let output = ctx.run(input, |ctx| {
                    crate::ui::draw_ui(
                        ctx,
                        &mut ui_scene,
                        &mut cam,
                        &mut ui_gizmo,
                        &mut offscreen_renderer,
                        &mut history,
                        &mut state,
                        None,
                    )
                });
                for (id, delta) in &output.textures_delta.set {
                    ui_renderer.update_texture(device, queue, *id, delta);
                }
                let r = state.viewport_rect;
                cam.aspect = r.width() / r.height().max(1.0);
                offscreen_renderer.viewport = Some([r.left(), r.top(), r.width(), r.height()]);
                ui_output = Some(output);
            }
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Screenshot Command Encoder"),
        });

        offscreen_renderer.render_scene(
            device,
            queue,
            &mut encoder,
            &color_view,
            &cam,
            if studio.is_some() { &ui_scene } else { scene },
            gizmo,
            None,
            None,
        );
        if let Some(output) = ui_output {
            let triangles = ctx.tessellate(output.shapes, output.pixels_per_point);
            let descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [w, h],
                pixels_per_point: output.pixels_per_point,
            };
            ui_renderer.update_buffers(device, queue, &mut encoder, &triangles, &descriptor);
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Studio screenshot UI"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            ui_renderer.render(&mut pass.forget_lifetime(), &triangles, &descriptor);
        }

        let unpadded_bytes_per_row = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = (padded_bytes_per_row * h) as wgpu::BufferAddress;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Screenshot Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| format!("Failed to receive buffer map signal: {e}"))?
            .map_err(|e| format!("Failed to map staging buffer: {e}"))?;

        let data = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for row in 0..h {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + (unpadded_bytes_per_row as usize);
            rgba.extend_from_slice(&data[start..end]);
        }
        drop(data);
        staging_buffer.unmap();

        let img = image::RgbaImage::from_raw(w, h, rgba)
            .ok_or_else(|| "Failed to construct RGBA image buffer".to_string())?;
        img.save(path)
            .map_err(|e| format!("Failed to save PNG screenshot to {}: {e}", path.display()))?;
        Ok(())
    }
}

/// Standalone headless renderer that initializes a wgpu device and renders the scene to a PNG file.
#[allow(clippy::too_many_arguments)]
pub fn render_scene_headless_to_png(
    camera: &OrbitCamera,
    scene: &Scene,
    shading: ViewportShading,
    show_wireframe: bool,
    show_grid: bool,
    width: u32,
    height: u32,
    path: &std::path::Path,
) -> Result<(), String> {
    render_headless(
        camera,
        scene,
        shading,
        show_wireframe,
        show_grid,
        width,
        height,
        path,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn render_headless(
    camera: &OrbitCamera,
    scene: &Scene,
    shading: ViewportShading,
    show_wireframe: bool,
    show_grid: bool,
    width: u32,
    height: u32,
    path: &std::path::Path,
    studio: Option<usize>,
    compare_split: Option<f32>,
) -> Result<(), String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "Failed to find GPU adapter for headless screenshot".to_string())?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Headless Screenshot Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .map_err(|e| format!("Failed to create headless wgpu device: {e}"))?;

    let mut renderer = Renderer3D::new(
        &device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        width.max(64),
        height.max(64),
    );
    renderer.shading = shading;
    renderer.show_wireframe_overlay = show_wireframe;
    renderer.show_grid = show_grid;
    renderer.compare_split = compare_split;

    let gizmo = GizmoState::default();
    renderer.render_to_png_file(
        &device, &queue, camera, scene, &gizmo, width, height, path, studio,
    )
}
