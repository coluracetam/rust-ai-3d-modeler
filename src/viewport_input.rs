//! Route input to the 3D workspace without treating its transparent egui panel as a control.
use crate::camera::OrbitCamera;
use glam::Vec2;
use winit::event::MouseScrollDelta;

pub(crate) fn accepts_pointer(
    ctx: &egui::Context,
    viewport: egui::Rect,
    window_pos: Vec2,
    dialog_open: bool,
) -> bool {
    let pos = egui::pos2(
        window_pos.x / ctx.pixels_per_point(),
        window_pos.y / ctx.pixels_per_point(),
    );
    !dialog_open
        && viewport.contains(pos)
        && !ctx.is_using_pointer()
        && !ctx.memory(|m| m.any_popup_open() || m.top_modal_layer().is_some())
        && ctx
            .layer_id_at(pos)
            .is_some_and(|layer| layer.order == egui::Order::Background)
}

pub(crate) fn scroll_camera(
    ctx: &egui::Context,
    camera: &mut OrbitCamera,
    viewport: egui::Rect,
    window_pos: Vec2,
    delta: MouseScrollDelta,
    dialog_open: bool,
) -> bool {
    if !accepts_pointer(ctx, viewport, window_pos, dialog_open) {
        return false;
    }
    let lines = match delta {
        MouseScrollDelta::LineDelta(_, y) => y,
        // Winit reports physical pixels; keep trackpad speed consistent across displays.
        MouseScrollDelta::PixelDelta(p) => p.y as f32 / ctx.pixels_per_point() / 20.0,
    };
    if !lines.is_finite() || lines == 0.0 {
        return false;
    }
    camera.zoom(lines);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::dpi::PhysicalPosition;

    // Exercise egui's actual panel ownership, not a mocked `consumed` flag.
    fn workspace(scale: f32, pointer: egui::Pos2) -> (egui::Context, egui::Rect) {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(scale);
        let mut viewport = egui::Rect::NOTHING;
        for frame in 0..3 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1000.0, 700.0),
                )),
                time: Some(frame as f64 / 60.0),
                events: vec![egui::Event::PointerMoved(pointer)],
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::TopBottomPanel::top("header")
                    .exact_height(80.0)
                    .show(ctx, |ui| {
                        ui.label("Project");
                    });
                egui::SidePanel::left("parts")
                    .exact_width(200.0)
                    .show(ctx, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for i in 0..100 {
                                ui.label(format!("Part {i}"));
                            }
                        });
                    });
                egui::SidePanel::right("review")
                    .exact_width(200.0)
                    .show(ctx, |ui| {
                        ui.label("Comments");
                    });
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ctx, |ui| {
                        viewport = ui.max_rect();
                        egui::Area::new("camera_controls".into())
                            .fixed_pos(egui::pos2(250.0, 120.0))
                            .show(ctx, |ui| {
                                egui::Frame::window(ui.style()).show(ui, |ui| {
                                    ui.set_min_size(egui::vec2(200.0, 60.0));
                                    let _ = ui.button("Camera");
                                });
                            });
                    });
            });
        }
        (ctx, viewport)
    }

    #[test]
    fn wheel_zooms_empty_central_panel_even_when_egui_wants_pointer() {
        let pointer = egui::pos2(600.0, 450.0);
        let (ctx, viewport) = workspace(1.0, pointer);
        assert!(
            ctx.wants_pointer_input(),
            "reproduce the transparent panel claiming the pointer"
        );
        for orthographic in [false, true] {
            let mut camera = OrbitCamera {
                orthographic,
                ..Default::default()
            };
            let before = camera.distance;
            assert!(scroll_camera(
                &ctx,
                &mut camera,
                viewport,
                Vec2::new(pointer.x, pointer.y),
                MouseScrollDelta::LineDelta(0.0, 1.0),
                false
            ));
            assert!(camera.distance < before);
            let closer = camera.distance;
            assert!(scroll_camera(
                &ctx,
                &mut camera,
                viewport,
                Vec2::new(pointer.x, pointer.y),
                MouseScrollDelta::LineDelta(0.0, -1.0),
                false
            ));
            assert!(camera.distance > closer);
        }
    }

    #[test]
    fn panels_controls_and_dialogs_keep_their_scroll_events() {
        for pointer in [
            egui::pos2(100.0, 400.0),
            egui::pos2(900.0, 400.0),
            egui::pos2(600.0, 40.0),
            egui::pos2(300.0, 150.0),
        ] {
            let (ctx, viewport) = workspace(1.0, pointer);
            let mut camera = OrbitCamera::default();
            let before = camera.distance;
            assert!(
                !scroll_camera(
                    &ctx,
                    &mut camera,
                    viewport,
                    Vec2::new(pointer.x, pointer.y),
                    MouseScrollDelta::LineDelta(0.0, 1.0),
                    false
                ),
                "UI at {pointer:?} must keep its wheel input"
            );
            assert_eq!(camera.distance, before);
        }
        let (ctx, viewport) = workspace(1.0, egui::pos2(600.0, 450.0));
        let mut camera = OrbitCamera::default();
        assert!(!scroll_camera(
            &ctx,
            &mut camera,
            viewport,
            Vec2::new(600.0, 450.0),
            MouseScrollDelta::LineDelta(0.0, 1.0),
            true
        ));
        ctx.memory_mut(|m| m.open_popup(egui::Id::new("project_menu")));
        assert!(!scroll_camera(
            &ctx,
            &mut camera,
            viewport,
            Vec2::new(600.0, 450.0),
            MouseScrollDelta::LineDelta(0.0, 1.0),
            false
        ));
    }

    #[test]
    fn trackpad_zoom_and_hit_testing_use_display_scale() {
        let mut distances = Vec::new();
        for scale in [1.0, 2.0] {
            let (ctx, viewport) = workspace(scale, egui::pos2(600.0, 450.0));
            let mut camera = OrbitCamera::default();
            assert!(scroll_camera(
                &ctx,
                &mut camera,
                viewport,
                Vec2::new(600.0 * scale, 450.0 * scale),
                MouseScrollDelta::PixelDelta(PhysicalPosition::new(0.0, 20.0 * scale as f64)),
                false
            ));
            distances.push(camera.distance);
            assert!(!accepts_pointer(
                &ctx,
                viewport,
                Vec2::new(100.0 * scale, 450.0 * scale),
                false
            ));
        }
        assert!((distances[0] - distances[1]).abs() < 1e-6);
    }

    #[test]
    fn fresh_cursor_position_routes_before_the_next_ui_frame() {
        let (ctx, viewport) = workspace(1.0, egui::pos2(100.0, 450.0));
        assert!(ctx.wants_pointer_input());
        assert!(accepts_pointer(
            &ctx,
            viewport,
            Vec2::new(600.0, 450.0),
            false
        ));
        let mut camera = OrbitCamera::default();
        assert!(!scroll_camera(
            &ctx,
            &mut camera,
            viewport,
            Vec2::new(600.0, 450.0),
            MouseScrollDelta::PixelDelta(PhysicalPosition::new(40.0, 0.0)),
            false
        ));
        assert_eq!(camera.distance, OrbitCamera::default().distance);
    }
}
