use dotenv::dotenv;
use eframe::{App, CreationContext};
use egui::Context;
use egui_graphs::{
    DefaultEdgeShape, DefaultNodeShape, Graph, GraphView, SettingsInteraction, SettingsNavigation,
    SettingsStyle,
};
use llmca::{
    system::{CognitiveSpace, MessageModelRule, VonNeumannLatticeCognitiveSpace},
    unit::CognitiveUnit,
};
use petgraph::Undirected;

pub struct BasicApp {
    g: Graph<CognitiveUnit, (), Undirected>,
}

impl BasicApp {
    fn new(_: &CreationContext<'_>, space: CognitiveSpace<MessageModelRule>) -> Self {
        let g = space.generate_graph();

        Self {
            g: Graph::<CognitiveUnit, (), Undirected>::from(&g),
        }
    }
}

impl App for BasicApp {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                &mut GraphView::<_, _, _, _, DefaultNodeShape, DefaultEdgeShape>::new(&mut self.g)
                    .with_styles(&SettingsStyle::new().with_labels_always(true))
                    .with_navigations(
                        &SettingsNavigation::default()
                            .with_fit_to_screen_enabled(false)
                            .with_zoom_and_pan_enabled(true),
                    )
                    .with_interactions(
                        &SettingsInteraction::new()
                            .with_dragging_enabled(true)
                            .with_node_clicking_enabled(true)
                            .with_node_selection_enabled(true),
                    ),
            );
        });
    }
}

fn main() {
    dotenv().ok();

    let rule = MessageModelRule::new("Hello, world!".to_string());
    let space = VonNeumannLatticeCognitiveSpace::new_lattice(3, 3, rule);

    println!("{:?}", space);

    let native_options = eframe::NativeOptions::default();

    eframe::run_native(
        "egui_graphs_basic_demo",
        native_options,
        Box::new(|cc| Box::new(BasicApp::new(cc, space))),
    )
    .unwrap();
}
