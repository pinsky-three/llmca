use std::{path::PathBuf, time};

use dotenv::dotenv;
use itertools::Itertools;

use llmca::system::{MessageModelRule, VonNeumannLatticeCognitiveSpace};

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

    let (n, m) = (20, 20);

    let rule_text = "You're a pixel in a image where are showing a summer sunset.
        The most important think is try to be stable and don't change your color too much.
        You only know the color of your neighbors and you need to choose your next color based on 
        the color of your neighbors and your itself. Always choose your next_state as hex color in
        a sequence (e.g. [\"00ff00\"])."
        .to_string();

    let rule = MessageModelRule::new(rule_text.clone());

    let initial_states = ["#ff0000", "#00ff00", "#0000ff"]
        .map(|d| d.to_string())
        .to_vec();

    let mut space = VonNeumannLatticeCognitiveSpace::new(rule, initial_states).build_lattice(n, m);

    let mut step = 0;

    let hash = md5::compute(rule_text.as_bytes());
    let hash_string = format!("{:x}", hash);

    let timestamp = time::SystemTime::now()
        .duration_since(time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let folder_name = PathBuf::from("saves")
        .join(hash_string)
        .join(format!("{}", timestamp));

    std::fs::create_dir_all(folder_name.clone()).unwrap();

    loop {
        println!("\nstep: {}", step);

        let all_states = space
            .get_units()
            .iter()
            .map(|u| serde_json::to_string(&u.state).unwrap())
            .collect::<Vec<_>>();

        let unique_states = all_states.iter().collect::<std::collections::HashSet<_>>();

        let states_to_colors = unique_states
            .iter()
            .sorted()
            // .enumerate()
            // .map(|(_i, state)| (state, get_color_from_hex_string(state)))
            .map(|state| (state, get_color_from_hex_string(state)))
            .collect::<std::collections::HashMap<_, _>>();

        println!(
            "states: {:?}",
            states_to_colors.keys().sorted().collect::<Vec<_>>()
        );

        space.get_units().iter().for_each(|unit| {
            let state = &serde_json::to_string(&unit.state).unwrap();

            let (p_x, p_y) = unit.position;

            let color = states_to_colors.get(&state).unwrap();

            let cell_size = (screen_width() / n as f32).min(screen_height() / m as f32);

            draw_rectangle(
                p_x as f32 * cell_size,
                p_y as f32 * cell_size,
                cell_size,
                cell_size,
                *color,
            );
        });

        let screen_image = get_screen_data().bytes;

        image::save_buffer(
            folder_name.join(format!("{}.png", step)),
            &screen_image,
            screen_width() as u32,
            screen_height() as u32,
            image::ColorType::Rgba8,
        )
        .unwrap();

        next_frame().await;

        space.sync_step();

        step += 1;
    }
}

// fn get_color_by_index(index: usize) -> Color {
//     [
//         LIGHTGRAY, GRAY, DARKGRAY, YELLOW, GOLD, ORANGE, PINK, RED, MAROON, GREEN, LIME, DARKGREEN,
//         SKYBLUE, BLUE, DARKBLUE, PURPLE, VIOLET, DARKPURPLE, BEIGE, BROWN, DARKBROWN, WHITE, BLACK,
//         BLANK, MAGENTA,
//     ][index % 25]
// }

fn get_color_from_hex_string(hex: &str) -> Color {
    let hex = hex.trim_matches(&['#', '"', '[', ']']).to_lowercase();

    if hex.len() != 6 {
        return Color::new(0.0, 0.0, 0.0, 1.0);
    }

    let r = u8::from_str_radix(&hex[0..2], 16).unwrap();
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap();
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap();

    Color::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
}
