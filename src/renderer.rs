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
}

struct CachedGpuMesh {
    key: MeshCacheKey,
    tri_buffer: Option<(wgpu::Buffer, u32)>,
    wire_buffer: Option<(wgpu::Buffer, u32)>,
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
    pub global_buffer: wgpu::Buffer,
    pub global_bind_group: wgpu::BindGroup,
    pub object_bind_group_layout: wgpu::BindGroupLayout,
    pub identity_line_bind_group: wgpu::BindGroup,
    pub depth_texture_view: wgpu::TextureView,
    pub depth_size: (u32, u32),
    pub grid_vertex_buffer: wgpu::Buffer,
    pub grid_vertex_count: u32,
    pub ghost_unselected: bool,
    pub show_grid: bool,
    pub show_wireframe_overlay: bool,
    pub shading: ViewportShading,
    pub viewport: Option<[f32; 4]>,
    mesh_cache: HashMap<u64, CachedGpuMesh>,
}

impl Renderer3D {
    pub fn invalidate_cache(&mut self) {
        self.mesh_cache.clear();
    }

    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

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
        };
        let identity_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Identity Line Object Uniform Buffer"),
            contents: bytemuck::cast_slice(&[identity_u]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let identity_line_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Identity Line Bind Group"),
            layout: &object_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: identity_buf.as_entire_binding(),
            }],
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
            multisample: wgpu::MultisampleState::default(),
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
            multisample: wgpu::MultisampleState::default(),
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
            multisample: wgpu::MultisampleState::default(),
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
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let w = width.max(1);
        let h = height.max(1);
        let depth_texture_view = Self::create_depth_view(device, w, h);
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
            global_buffer,
            global_bind_group,
            object_bind_group_layout,
            identity_line_bind_group,
            depth_texture_view,
            depth_size: (w, h),
            grid_vertex_buffer,
            grid_vertex_count,
            show_grid: true,
            ghost_unselected: false,
            show_wireframe_overlay: false,
            shading: ViewportShading::StudioLit,
            viewport: None,
            mesh_cache: HashMap::new(),
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.depth_size != (w, h) {
            self.depth_texture_view = Self::create_depth_view(device, w, h);
            self.depth_size = (w, h);
        }
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
            sample_count: 1,
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
    ) -> wgpu::BindGroup {
        let normal_mat = model.inverse().transpose();
        let u = ObjectUniforms {
            model: model.to_cols_array_2d(),
            normal_mat: normal_mat.to_cols_array_2d(),
            params: [roughness, metallic, shading_mode, pattern_id],
            inspection: [opacity, 0.0, 0.0, 0.0],
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
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        })
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
        )
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
        let display = scene.display_scene();
        let scene = display.as_ref();
        let eye = camera.eye();
        let global_u = GlobalUniforms {
            view_proj: camera.view_proj_matrix().to_cols_array_2d(),
            camera_pos: [eye.x, eye.y, eye.z, 1.0],
            clip_plane: scene.clip_plane.unwrap_or([0.0, 0.0, 0.0, 1.0]),
        };
        queue.write_buffer(&self.global_buffer, 0, bytemuck::cast_slice(&[global_u]));

        // Update per-object GPU mesh caches only if dirty
        let is_edit_mode = matches!(scene.mode, InteractionMode::EditMesh(_));
        for (i, obj) in scene.objects.iter().enumerate() {
            if !obj.visible {
                continue;
            }
            let is_selected = scene.selected_object == Some(i);
            let has_face_sel = !obj.mesh.selected_faces.is_empty() || (is_selected && is_edit_mode);
            let key = MeshCacheKey {
                mesh_version: obj.mesh.version,
                color_bits: [
                    obj.color[0].to_bits(),
                    obj.color[1].to_bits(),
                    obj.color[2].to_bits(),
                ],
                flat_shading: obj.flat_shading,
                is_selected,
                is_edit_mode: has_face_sel,
            };

            let needs_rebuild = self
                .mesh_cache
                .get(&obj.id)
                .map_or(true, |cached| cached.key != key);

            if needs_rebuild {
                let tris = obj.mesh.build_gpu_triangles(obj.color, obj.flat_shading);
                let tri_buffer = if !tris.is_empty() {
                    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Cached Mesh Tri VB"),
                        contents: bytemuck::cast_slice(&tris),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    Some((vb, tris.len() as u32))
                } else {
                    None
                };

                let base_wire_col = if has_face_sel {
                    [0.18, 0.22, 0.28]
                } else if is_selected {
                    [0.95, 0.62, 0.18]
                } else {
                    [0.32, 0.35, 0.40]
                };
                // Build wireframe in local space (Mat4::IDENTITY) so object transform is applied on GPU
                let lines =
                    obj.mesh
                        .build_wireframe_lines(Mat4::IDENTITY, base_wire_col, has_face_sel);
                let wire_buffer = if !lines.is_empty() {
                    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Cached Mesh Wire VB"),
                        contents: bytemuck::cast_slice(&lines),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    Some((vb, lines.len() as u32))
                } else {
                    None
                };

                self.mesh_cache.insert(
                    obj.id,
                    CachedGpuMesh {
                        key,
                        tri_buffer,
                        wire_buffer,
                    },
                );
            }
        }

        // Collect per-object uniform bind groups
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

        let mut ghost_draws: Vec<(u64, wgpu::BindGroup, f32)> = Vec::new();
        let mut solid_draws: Vec<(u64, wgpu::BindGroup)> = Vec::new();
        let mut wire_draws: Vec<(u64, wgpu::BindGroup)> = Vec::new();
        let mut depth_line_draws: Vec<PreparedDrawCall> = Vec::new();
        let mut overlay_line_draws: Vec<PreparedDrawCall> = Vec::new();

        for (i, obj) in scene.objects.iter().enumerate() {
            if !obj.visible {
                continue;
            }
            let is_selected = scene.selected_object == Some(i);
            let has_face_sel = !obj.mesh.selected_faces.is_empty() || (is_selected && is_edit_mode);
            let model = obj.transform.matrix();

            if self.shading != ViewportShading::WireframeOnly {
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
                    if self.ghost_unselected
                        && scene.has_any_selection()
                        && !is_selected
                        && !has_face_sel
                    {
                        0.16
                    } else {
                        1.0
                    },
                );
                if self.ghost_unselected
                    && scene.has_any_selection()
                    && !is_selected
                    && !has_face_sel
                {
                    ghost_draws.push((
                        obj.id,
                        bg,
                        (obj.transform.translation - eye).length_squared(),
                    ));
                } else {
                    solid_draws.push((obj.id, bg));
                }
            }

            let want_wire = self.shading == ViewportShading::WireframeOnly
                || self.show_wireframe_overlay
                || has_face_sel;

            if want_wire {
                let bg = self.create_line_bind_group(device, model);
                wire_draws.push((obj.id, bg));
            }

            if is_selected
                && matches!(
                    scene.mode,
                    InteractionMode::EditMesh(EditElementMode::Vertex)
                )
            {
                let mut v_lines = Vec::new();
                let r = 0.035;
                for (vi, &p) in obj.mesh.positions.iter().enumerate() {
                    let wp = model.transform_point3(p);
                    let sel = obj.mesh.selected_vertices.contains(&vi);
                    let col = if sel {
                        [1.0, 0.72, 0.1]
                    } else {
                        [0.45, 0.52, 0.62]
                    };
                    let s = if sel { r * 1.4 } else { r * 0.75 };
                    for axis in [Vec3::X * s, Vec3::Y * s, Vec3::Z * s] {
                        v_lines.push(GpuVertex {
                            position: (wp - axis).to_array(),
                            normal: [0.0, 1.0, 0.0],
                            color: col,
                        });
                        v_lines.push(GpuVertex {
                            position: (wp + axis).to_array(),
                            normal: [0.0, 1.0, 0.0],
                            color: col,
                        });
                    }
                }
                if !v_lines.is_empty() {
                    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Vertex Cross VB"),
                        contents: bytemuck::cast_slice(&v_lines),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let bg = self.create_line_bind_group(device, Mat4::IDENTITY);
                    overlay_line_draws.push(PreparedDrawCall {
                        vertex_buffer: vb,
                        vertex_count: v_lines.len() as u32,
                        bind_group: bg,
                    });
                }
            }
        }

        // Live Hover Preview Wireframe (Smart Sub-Part / Smart Surface highlight before clicking)
        if let Some((hover_obj_id, hover_faces)) = hover_preview {
            if !hover_faces.is_empty() {
                if let Some(obj) = scene
                    .objects
                    .iter()
                    .find(|o| o.id == hover_obj_id && o.visible)
                {
                    let preview_lines = obj.mesh.build_face_subset_lines(
                        hover_faces,
                        obj.transform.matrix(),
                        [0.15, 0.94, 1.0],
                    );
                    if !preview_lines.is_empty() {
                        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Hover Preview Wireframe VB"),
                            contents: bytemuck::cast_slice(&preview_lines),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                        let bg = self.create_line_bind_group(device, Mat4::IDENTITY);
                        depth_line_draws.push(PreparedDrawCall {
                            vertex_buffer: vb,
                            vertex_count: preview_lines.len() as u32,
                            bind_group: bg,
                        });
                    }
                }
            }
        }

        // 3D Spatial Comment Pin & Region Markers (for AI-driven review workflow)
        let mut pin_lines = Vec::new();
        for c in &scene.comments {
            if c.resolved || c.is_general || c.object_id == 0 || scene.comment_needs_reattachment(c)
            {
                continue;
            }
            let is_active = scene.active_comment_id == Some(c.id);
            let col = if is_active {
                [0.18, 0.95, 1.0]
            } else {
                [1.0, 0.52, 0.18]
            };
            let r = c.selection_radius.clamp(0.15, 2.5);
            if !c.targets.is_empty() {
                for t in &c.targets {
                    pin_lines.extend(Self::build_comment_pin_lines(
                        scene
                            .refreshed_target(t)
                            .map(|t| t.world_anchor)
                            .unwrap_or(t.world_anchor),
                        scene
                            .refreshed_target(t)
                            .map(|t| t.surface_normal)
                            .unwrap_or(t.surface_normal),
                        r,
                        col,
                    ));
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

        // Draw green 3D selection pins for all currently selected sections across the scene
        let mut drew_section_pin = false;
        for obj in &scene.objects {
            if !obj.visible || obj.mesh.selected_faces.is_empty() {
                continue;
            }
            if let Some((lc, ln, _min_b, _max_b)) = obj.mesh.selected_faces_bounds_and_normal() {
                let model = obj.transform.matrix();
                let wp = model.transform_point3(lc);
                let wn = model
                    .inverse()
                    .transpose()
                    .transform_vector3(ln)
                    .normalize_or_zero();
                pin_lines.extend(Self::build_comment_pin_lines(
                    wp,
                    wn,
                    0.35,
                    [0.25, 1.0, 0.65],
                ));
                drew_section_pin = true;
            }
        }
        if !drew_section_pin {
            if let Some((_obj_id, wp, _lp, wn)) = scene.pending_anchor {
                pin_lines.extend(Self::build_comment_pin_lines(
                    wp,
                    wn,
                    0.35,
                    [0.25, 1.0, 0.65],
                ));
            }
        }
        if !pin_lines.is_empty() {
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Comment Pins VB"),
                contents: bytemuck::cast_slice(&pin_lines),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let bg = self.create_line_bind_group(device, Mat4::IDENTITY);
            overlay_line_draws.push(PreparedDrawCall {
                vertex_buffer: vb,
                vertex_count: pin_lines.len() as u32,
                bind_group: bg,
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
                let bg = self.create_line_bind_group(device, Mat4::IDENTITY);
                overlay_line_draws.push(PreparedDrawCall {
                    vertex_buffer: vb,
                    vertex_count: ring_lines.len() as u32,
                    bind_group: bg,
                });
            }
        }

        // Execute Render Pass
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("3D Viewport Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.025,
                        g: 0.035,
                        b: 0.052,
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
        // 1. Solid Meshes (using cached GPU vertex buffers)
        if !solid_draws.is_empty() {
            rpass.set_pipeline(&self.triangle_pipeline);
            rpass.set_bind_group(0, &self.global_bind_group, &[]);
            for (obj_id, bg) in &solid_draws {
                if let Some(cached) = self.mesh_cache.get(obj_id) {
                    if let Some((vb, count)) = &cached.tri_buffer {
                        rpass.set_bind_group(1, bg, &[]);
                        rpass.set_vertex_buffer(0, vb.slice(..));
                        rpass.draw(0..*count, 0..1);
                    }
                }
            }
        }

        ghost_draws.sort_by(|a, b| b.2.total_cmp(&a.2));
        rpass.set_pipeline(&self.ghost_pipeline);
        rpass.set_bind_group(0, &self.global_bind_group, &[]);
        for (id, bg, _) in &ghost_draws {
            if let Some((vb, count)) = self.mesh_cache.get(id).and_then(|m| m.tri_buffer.as_ref()) {
                rpass.set_bind_group(1, bg, &[]);
                rpass.set_vertex_buffer(0, vb.slice(..));
                rpass.draw(0..*count, 0..1);
            }
        }

        // 2. Depth-tested Lines (Preallocated Grid + Cached Mesh Wireframes + Hover Preview)
        if (self.show_grid && self.grid_vertex_count > 0)
            || !wire_draws.is_empty()
            || !depth_line_draws.is_empty()
        {
            rpass.set_pipeline(&self.line_depth_pipeline);
            rpass.set_bind_group(0, &self.global_bind_group, &[]);

            if self.show_grid && self.grid_vertex_count > 0 {
                rpass.set_bind_group(1, &self.identity_line_bind_group, &[]);
                rpass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
                rpass.draw(0..self.grid_vertex_count, 0..1);
            }

            for (obj_id, bg) in &wire_draws {
                if let Some(cached) = self.mesh_cache.get(obj_id) {
                    if let Some((vb, count)) = &cached.wire_buffer {
                        rpass.set_bind_group(1, bg, &[]);
                        rpass.set_vertex_buffer(0, vb.slice(..));
                        rpass.draw(0..*count, 0..1);
                    }
                }
            }

            for call in &depth_line_draws {
                rpass.set_bind_group(1, &call.bind_group, &[]);
                rpass.set_vertex_buffer(0, call.vertex_buffer.slice(..));
                rpass.draw(0..call.vertex_count, 0..1);
            }
        }

        // 3. Overlay Lines (Gizmo & Brush Cursor)
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
        offscreen_renderer.show_wireframe_overlay = self.show_wireframe_overlay;
        offscreen_renderer.show_grid = self.show_grid;

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
            let mut history = crate::history::UndoHistory::default();
            let mut ui_gizmo = GizmoState::default();
            for frame in 0..3 {
                let input = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(w as f32, h as f32),
                    )),
                    time: Some(frame as f64 / 60.0),
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

    let gizmo = GizmoState::default();
    renderer.render_to_png_file(
        &device, &queue, camera, scene, &gizmo, width, height, path, studio,
    )
}
