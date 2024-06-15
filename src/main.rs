use dotenv::dotenv;
use eframe::{App, CreationContext};
use egui::Context;
use egui_graphs::{DefaultEdgeShape, DefaultNodeShape, Graph, GraphView};
use llmca::system::{CognitiveSpace, MessageModelRule, VonNeumannLatticeCognitiveSpace};
use petgraph::Directed;

pub struct BasicApp {
    g: Graph<(), (), Directed>,
}

impl BasicApp {
    fn new(_: &CreationContext<'_>, space: CognitiveSpace<MessageModelRule>) -> Self {
        let g = space.generate_graph();
        Self { g: Graph::from(&g) }
    }
}

impl App for BasicApp {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(&mut GraphView::<
                _,
                _,
                _,
                _,
                DefaultNodeShape,
                DefaultEdgeShape,
            >::new(&mut self.g));
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
