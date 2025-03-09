#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use dynamical_system::life::{entity::Entity, manager::LifeManager};
use eframe::egui::{self, Frame, Sense, Slider, UiBuilder};

fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1024.0, 1024.0]),
        ..Default::default()
    };
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| {
            // This gives us image support:
            // egui_extras::install_image_loaders(&cc.egui_ctx);

            Ok(Box::<MyApp>::default())
        }),
    )
}

#[derive(Default)]
struct MyApp {
    life_manager: LifeManager,
    selected_entity: usize,
    loaded_entity: Option<Entity>,
    current_step: usize,
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open").clicked() {
                        // …
                    }
                });
            });

            ui.heading("Life Manager");

            let entities = self.life_manager.list_entities();

            egui::ComboBox::from_label("Select an organism").show_index(
                ui,
                &mut self.selected_entity,
                entities.len(),
                |i| entities[i].clone(),
            );

            ui.horizontal(|ui| {
                // let name_label = ui.label("Your name: ");
                // ui.text_edit_singleline(&mut self.name)
                //     .labelled_by(name_label.id);
                if ui.button("load").clicked() {
                    self.loaded_entity = self
                        .life_manager
                        .get_entity(&entities[self.selected_entity])
                        .cloned();

                    self.current_step = self.loaded_entity.clone().unwrap().current_step() as usize;
                }
            });

            if self.loaded_entity.is_some() {
                ui.heading("Entity");

                let entity = self.loaded_entity.as_ref().unwrap();

                ui.label(format!("ID: {}", entity.id()));

                ui.label(format!("Step: {}", entity.current_step()));

                // ui.label(format!("Space: {:?}", entity.space()));

                let mut latest_p_y = 0;

                ui.add(Slider::new(
                    &mut self.current_step,
                    0..=(entity.current_step() as usize),
                ));

                egui::Grid::new("some_unique_id")
                    .spacing([0.0, 0.0])
                    .show(ui, |ui| {
                        entity
                            .space_at(self.current_step)
                            .get_units()
                            .iter()
                            .for_each(|unit| {
                                let state = &unit.memory.last().unwrap().state;

                                let (p_y, p_x) = unit.position;

                                if latest_p_y != p_y {
                                    // println!("change of p_y: {}", p_y);
                                    ui.end_row();
                                }

                                latest_p_y = p_y;

                                let color = state.clone();

                                let cell_size = (1024.0f32 / 10.0).min(720.0 / 10.0);
                                // let cell = [p_x as f32 * cell_size, p_y as f32 * cell_size];

                                let response = ui
                                    .scope_builder(
                                        UiBuilder::new()
                                            .id_salt("interactive_container")
                                            .sense(Sense::click() & Sense::hover()),
                                        |ui| {
                                            let response = ui.response();
                                            let visuals = ui.style().interact(&response);
                                            // let text_color = visuals.text_color();

                                            Frame::canvas(ui.style())
                                                // .fill(visuals.bg_fill.gamma_multiply(0.3))
                                                .fill(
                                                    egui::Color32::from_hex(&color)
                                                        .unwrap_or(egui::Color32::BLACK),
                                                )
                                                .stroke(visuals.bg_stroke)
                                                // .inner_margin(ui.spacing().menu_margin)
                                                .show(ui, |ui| {
                                                    ui.set_width(cell_size);
                                                    ui.set_height(cell_size);

                                                    // ui.add_space(16.0);
                                                    // ui.vertical_centered(|ui| {
                                                    //     Label::new(
                                                    //         RichText::new(state.to_string())
                                                    //             .color(text_color)
                                                    //             .size(16.0),
                                                    //     )
                                                    //     .selectable(false)
                                                    //     .ui(ui);
                                                    // });
                                                    // ui.add_space(16.0);

                                                    // ui.vertical(|ui| {
                                                    // ui.with_layout(
                                                    //     egui::Layout::bottom_up(egui::Align::Max),
                                                    //     |ui| {
                                                    //         if ui.button("update").clicked() {
                                                    //             // self.count = 0;
                                                    //         }
                                                    //     },
                                                    // );
                                                    // });
                                                });
                                        },
                                    )
                                    .response
                                    .on_hover_text(state.to_string());

                                if response.clicked() {
                                    // self.count += 1;
                                }

                                // if response.hovered() {
                                //     ui.label(format!("state: {}", state));
                                // }
                            });
                    });
            }
            // ui.add(egui::Slider::new(&mut self.age, 0..=120).text("age"));
            // if ui.button("Increment").clicked() {
            //     self.age += 1;
            // }
            // ui.label(format!("Hello '{}', age {}", self.name, self.age));

            // ui.image(egui::include_image!(
            //     "../../../crates/egui/assets/ferris.png"
            // ));
        });
    }
}
