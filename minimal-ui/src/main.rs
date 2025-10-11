use dotenv::dotenv;
use dynamical_system::{
    life::{entity::Entity, manager::LifeManager},
    system::unit_next::CognitiveUnitPair,
};
use itertools::Itertools;
use macroquad::prelude::*;

fn window_conf() -> Conf {
    Conf {
        window_title: "LLMCA".to_owned(),
        window_width: 800,
        window_height: 800,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    dotenv().ok();

    let rt = tokio::runtime::Runtime::new().unwrap();

    let manager = LifeManager::default();

    let size = (20, 20);

    let initial_state = (0..size.0)
        .cartesian_product(0..size.1)
        .map(|_p| CognitiveUnitPair {
            rule: "you're a pixel in a sunset video, update your state to create an emotive scene"
                .to_string(),
            state: "#bababa".to_string(),
        })
        .collect();

    let temporal_memory_size = 4;

    let mut entity = Entity::new_2d_lattice(&manager, initial_state, size, temporal_memory_size);

    loop {
        let unique_states = entity.calculate_unique_states();

        let states_to_colors = unique_states
            .iter()
            .sorted()
            .map(|state| (state, get_color_from_hex_string(state)))
            .collect::<std::collections::HashMap<_, _>>();

        println!(
            "states: {:?}",
            states_to_colors.keys().sorted().collect::<Vec<_>>()
        );

        entity.loaded_space().get_units().iter().for_each(|unit| {
            let state = &unit.memory.last().unwrap().state;

            let (p_x, p_y) = unit.position;

            let color = states_to_colors.get(&state).unwrap();

            let cell_size = (screen_width() / size.0 as f32).min(screen_height() / size.1 as f32);

            draw_rectangle(
                p_x as f32 * cell_size,
                p_y as f32 * cell_size,
                cell_size,
                cell_size,
                *color,
            );
        });

        next_frame().await;

        rt.block_on(async {
            entity.evolve_async().await;
        });
    }
}

fn get_color_from_hex_string(hex: &str) -> Color {
    let hex = hex.trim_matches(['#', '\"', '[', ']']).to_lowercase();

    if hex.len() < 6 {
        return Color::new(0.0, 0.0, 0.0, 1.0);
    }

    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);

    Color::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
}
