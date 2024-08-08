use std::{path::PathBuf, time};

use dotenv::dotenv;

use itertools::Itertools;

use llmca::system::{MessageModelRule, VonNeumannLatticeCognitiveSpace};

use macroquad::prelude::*;
use tokio::runtime::Runtime;

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

    let (n, m) = (30, 30);

    let rule_text = "You're a pixel in a video, you choose
    your color based on the color of your neighbors. and always as hex string: \"#ffffff\".
    You can also add a comment to explain your choice in your second channel.
    Your video is a representation of the rain, choose colors that represent the rain.
     (e.g. [\"#ffffff\", \"I select white because I'm part of a cloud\"])."
        .to_string();

    let rule = MessageModelRule::new(rule_text.clone(), vec![]);

    let initial_states = [vec![
        "#aaaaaa".to_string(),
        "Hello There, I'm using this channel to share internal thoughts.".to_string(),
    ]]
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

    let rt = Runtime::new().unwrap();

    loop {
        println!("\nstep: {}", step);

        let all_states = space
            .get_units()
            .iter()
            .map(|u| serde_json::to_string(&u.state.first()).unwrap())
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
            let state = &serde_json::to_string(&unit.state.first()).unwrap();

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

        rt.block_on(async {
            space.distributed_step().await;
        });

        step += 1;
    }
}

fn get_color_from_hex_string(hex: &str) -> Color {
    let hex = hex.trim_matches(&['#', '"', '[', ']']).to_lowercase();

    if hex.len() < 6 {
        return Color::new(0.0, 0.0, 0.0, 1.0);
    }

    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);

    Color::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
}
