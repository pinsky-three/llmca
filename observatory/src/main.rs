#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::sync::Arc;

use dotenvy::dotenv;
use dynamical_system::life::{
    entity::{Entity, EntityState},
    manager::LifeManager,
};
use eframe::egui::{self, CornerRadius, Frame, Margin, Sense, Slider, UiBuilder, Vec2};
use tokio::sync::Mutex;

fn main() -> eframe::Result {
    dotenv().ok();

    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1024.0, 1024.0]),
        ..Default::default()
    };
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| Ok(Box::<LifeManagerApp>::default())),
    )
}

// #[derive(Default)]
struct LifeManagerApp {
    life_manager: LifeManager,
    selected_entity: usize,
    loaded_entity: Option<ManagedEntity>,
    current_step: usize,
    runtime: tokio::runtime::Runtime,
    // worker_state: WorkerState,
}

// enum WorkerState {
//     Idle,
//     Evolving(ManagedEntity),
// }

#[derive(Clone)]
struct ManagedEntity {
    entity: Arc<tokio::sync::Mutex<Entity>>,
}

impl ManagedEntity {
    fn state(&self) -> EntityState {
        self.entity.blocking_lock().state().clone()
    }

    fn load_space_at(
        &self,
        step: usize,
    ) -> dynamical_system::system::space::CognitiveSpaceWithMemory {
        self.entity.blocking_lock().load_space_at(step)
    }

    fn evolve(&mut self, runtime: &tokio::runtime::Runtime) {
        let handle = runtime.handle().clone();
        let entity = self.entity.clone();

        runtime.spawn(async move {
            println!("a");
            let mut locked_entity = entity.lock().await;
            println!("b");
            locked_entity.evolve(&handle).await;
            println!("c");

            // done evolving
            // locked_entity.set_state(EntityState::Idle);
        });
    }
    fn current_step(&self) -> u32 {
        self.entity.blocking_lock().current_step()
    }

    fn loaded_space(&self) -> dynamical_system::system::space::CognitiveSpaceWithMemory {
        self.entity.blocking_lock().loaded_space().clone()
    }

    fn id(&self) -> String {
        self.entity.blocking_lock().id().to_string()
    }
}

impl Default for LifeManagerApp {
    fn default() -> Self {
        Self {
            life_manager: LifeManager::default(),
            selected_entity: 0,
            loaded_entity: None,
            current_step: 0,
            runtime: tokio::runtime::Runtime::new().unwrap(),
            // worker_state: WorkerState::Idle,
        }
    }
}

impl eframe::App for LifeManagerApp {
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
                    // let ent = ;
                    // .cloned();
                    self.loaded_entity = self
                        .life_manager
                        .get_entity(&entities[self.selected_entity])
                        .map(|e| ManagedEntity {
                            entity: Arc::new(Mutex::new(e.clone())),
                        });

                    self.current_step =
                        self.loaded_entity.as_mut().unwrap().current_step() as usize;
                }

                if let Some(managed_entity) = self.loaded_entity.as_mut() {
                    // let cloned_entity = managed_entity.clone();

                    if let EntityState::ComputingStep(_step) = managed_entity.state() {
                        ui.disable();
                    }

                    if ui.button("evolve").clicked() {
                        // self.worker_state = WorkerState::Evolving(&entity);
                        // self.worker_state = WorkerState::Evolving(cloned_entity);

                        managed_entity.evolve(&self.runtime);
                    }

                    if let EntityState::ComputingStep(step) = managed_entity.state() {
                        ui.label(format!("Computing step {}", step));

                        // let tasks = managed_entity.space_at(self.current_step).computing_tasks();
                        let tasks = managed_entity.loaded_space().computing_tasks();

                        println!("len tasks: {}", tasks.len());

                        // let total = tasks.first().unwrap().total_units;
                        let total = self
                            .loaded_entity
                            .as_ref()
                            .unwrap()
                            .loaded_space()
                            .get_units()
                            .len();

                        println!("len total: {}", total);

                        ui.add(
                            egui::ProgressBar::new(tasks.len() as f32 / total as f32)
                                .desired_width(100.0),
                        );
                    }
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

                let cell_size = 48.0;

                ui.spacing_mut().item_spacing = Vec2::new(0.0, 0.0);

                egui::Grid::new("entity")
                    .spacing(Vec2::new(0.0, 0.0))
                    .show(ui, |ui| {
                        entity
                            .load_space_at(self.current_step)
                            .get_units()
                            .iter()
                            .for_each(|unit| {
                                let state = &unit.memory.last().unwrap().state;
                                let rule = &unit.memory.last().unwrap().rule;

                                let (p_y, _p_x) = unit.position;

                                if latest_p_y != p_y {
                                    ui.end_row();
                                }

                                latest_p_y = p_y;

                                let color = state.clone();

                                let response = ui
                                    .scope_builder(
                                        UiBuilder::new()
                                            .id_salt("individual")
                                            .sense(Sense::click() & Sense::hover()),
                                        |ui| {
                                            Frame::canvas(ui.style())
                                                .fill(
                                                    egui::Color32::from_hex(&color)
                                                        .unwrap_or(egui::Color32::BLACK),
                                                )
                                                .corner_radius(CornerRadius::ZERO)
                                                .inner_margin(Margin::ZERO)
                                                .outer_margin(Margin::ZERO)
                                                .show(ui, |ui| {
                                                    ui.set_width(cell_size);
                                                    ui.set_height(cell_size);
                                                });
                                        },
                                    )
                                    .response
                                    .on_hover_text(format!(
                                        "pos: {:?}\nstate: {}\nrule: {}",
                                        unit.position, state, rule,
                                    ));

                                if response.clicked() {
                                    // self.count += 1;
                                }
                            });
                    });

                // if ui.button("evolve").clicked() {
                //     self.loaded_entity.as_mut().unwrap().evolve();
                //     // self.runtime.block_on(async {
                //     // });
                // }
            }
        });
    }
}
