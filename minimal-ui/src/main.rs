//! Minimal UI for the programmable-cell PoC.
//!
//! Per tick, every cell asks the LLM for a fresh `CellProgram` (a tiny Python
//! `step` function) given its current color, neighbor colors and a stable rule.
//! The returned program is validated by `monty`, executed locally to compute
//! the next color, and persisted to disk so you can read what the model wrote.
//!
//! Calls are sharded across all configured resolvers (mirrors the
//! `distributed_step` pattern from `main`). The window blocks while a tick is
//! in flight — exactly like the `main`-branch behavior.

use dotenv::dotenv;
use dynamical_system::system::{
    programmable::{propose_program_via_resolver, CellProgram, Rgb},
    space::{load_llm_resolvers, LLMResolver},
};
use itertools::Itertools;
use macroquad::prelude::*;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing_subscriber::EnvFilter;

const GRID_SIZE: (usize, usize) = (3, 3);
const DEFAULT_RULE: &str =
    "spreading sunset: blend with neighbors and slowly drift toward warm colors";
/// Where per-cell programs are dumped. One file per (step, x, y).
const PROGRAMS_DIR: &str = ".minimal_ui_programs";

fn window_conf() -> Conf {
    Conf {
        window_title: "LLMCA — programmable cells".to_owned(),
        window_width: 800,
        window_height: 800,
        ..Default::default()
    }
}

#[derive(Clone)]
struct Cell {
    /// Stable per-cell rule the model is asked to honor across ticks.
    rule: String,
    program: CellProgram,
    color: Rgb,
}

impl Cell {
    fn new_with_rule(rule: String, color: Rgb) -> Self {
        Self {
            rule,
            program: CellProgram::identity(),
            color,
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    dotenv().ok();
    init_tracing();
    let startup_started_at = Instant::now();

    let rt = tokio::runtime::Runtime::new().unwrap();

    let (n, m) = GRID_SIZE;
    let mut cells: Vec<Cell> = (0..n)
        .cartesian_product(0..m)
        .map(|(x, y)| Cell::new_with_rule(DEFAULT_RULE.to_string(), seed_color((x, y), (n, m))))
        .collect();

    let llm_disabled = env_flag("LLMCA_NO_LLM");
    let resolvers: Vec<LLMResolver> = if llm_disabled {
        tracing::info!("minimal_ui_llm_disabled_using_local_demo_program");
        let demo = demo_program();
        for cell in &mut cells {
            cell.program = demo.clone();
        }
        vec![]
    } else {
        let r = load_llm_resolvers();
        if r.is_empty() {
            tracing::warn!(
                "no resolvers configured — populate ./resolvers.toml or set OPENAI_API_URL/MODEL_NAME/API_KEY. Falling back to local demo."
            );
            let demo = demo_program();
            for cell in &mut cells {
                cell.program = demo.clone();
            }
        }
        r
    };

    let dump_dir = PathBuf::from(PROGRAMS_DIR);
    if !dump_dir.exists() {
        let _ = std::fs::create_dir_all(&dump_dir);
    }

    tracing::info!(
        n,
        m,
        resolvers = resolvers.len(),
        dump_dir = %dump_dir.display(),
        startup_elapsed_ms = startup_started_at.elapsed().as_millis() as u64,
        "minimal_ui_ready"
    );

    let mut step_index: u32 = 0;

    loop {
        // Render current colors.
        let cell_size = (screen_width() / n as f32).min(screen_height() / m as f32);
        for x in 0..n {
            for y in 0..m {
                let c = cells[index(x, y, m)].color;
                draw_rectangle(
                    x as f32 * cell_size,
                    y as f32 * cell_size,
                    cell_size,
                    cell_size,
                    Color::new(
                        c.r as f32 / 255.0,
                        c.g as f32 / 255.0,
                        c.b as f32 / 255.0,
                        1.0,
                    ),
                );
            }
        }

        // Press R to nuke programs and let the LLM start fresh next tick.
        if is_key_pressed(KeyCode::R) {
            tracing::info!("minimal_ui_reset_requested");
            for cell in &mut cells {
                cell.program = CellProgram::identity();
            }
        }

        next_frame().await;

        let started_at = Instant::now();
        let frame_telemetry = if !resolvers.is_empty() {
            rt.block_on(llm_evolve_step(
                &mut cells,
                &resolvers,
                step_index,
                (n, m),
                Some(&dump_dir),
            ))
        } else {
            local_evolve_step(&mut cells, step_index, (n, m))
        };

        tracing::info!(
            step_index,
            llm_calls = frame_telemetry.llm_calls,
            llm_failures = frame_telemetry.llm_failures,
            run_failures = frame_telemetry.run_failures,
            elapsed_ms = started_at.elapsed().as_millis() as u64,
            "minimal_ui_evolved_frame"
        );

        step_index += 1;
    }
}

#[derive(Default)]
struct StepTelemetry {
    llm_calls: usize,
    llm_failures: usize,
    run_failures: usize,
}

/// Per-tick LLM-driven evolution. For each cell we ask the model for a fresh
/// `CellProgram` given its current color and neighbors, then run the returned
/// code locally to obtain the next color. Sharded across all resolvers.
async fn llm_evolve_step(
    cells: &mut [Cell],
    resolvers: &[LLMResolver],
    step_index: u32,
    (n, m): (usize, usize),
    dump_dir: Option<&Path>,
) -> StepTelemetry {
    let total = cells.len();
    let snapshot: Vec<Rgb> = cells.iter().map(|c| c.color).collect();
    let chunk_width = resolvers.len().max(1);

    // Per-cell next state, written from the joined tasks.
    let mut next: Vec<Option<(CellProgram, Rgb)>> = (0..total).map(|_| None).collect();
    let mut telemetry = StepTelemetry::default();

    for chunk_start in (0..total).step_by(chunk_width) {
        let chunk_end = (chunk_start + chunk_width).min(total);
        let mut tasks = Vec::with_capacity(chunk_end - chunk_start);

        for (i, cell_idx) in (chunk_start..chunk_end).enumerate() {
            let resolver = resolvers[i % resolvers.len()].clone();
            let (x, y) = (cell_idx / m, cell_idx % m);
            let neighbors_xy = moore_neighbors((x, y), (n, m));
            let neighbors_rgb: Vec<Rgb> =
                neighbors_xy.iter().map(|&(nx, ny)| snapshot[index(nx, ny, m)]).collect();
            let cell = cells[cell_idx].clone();
            let current_color = snapshot[cell_idx];

            let instruction = build_per_cell_instruction(
                &cell.rule,
                &cell.program,
                current_color,
                &neighbors_rgb,
                step_index,
            );

            tasks.push(tokio::spawn(async move {
                let result = propose_program_via_resolver(&resolver, &instruction).await;
                (cell_idx, x, y, current_color, neighbors_rgb, result)
            }));
        }

        for joined in futures::future::join_all(tasks).await {
            let (cell_idx, x, y, current_color, neighbors_rgb, result) = match joined {
                Ok(t) => t,
                Err(err) => {
                    tracing::warn!(error = %err, step_index, "llm_cell_task_join_failed");
                    telemetry.llm_failures += 1;
                    continue;
                }
            };
            telemetry.llm_calls += 1;
            match result {
                Ok(prog) => match prog.run(current_color, &neighbors_rgb, step_index) {
                    Ok(next_rgb) => {
                        log_and_dump_program(
                            dump_dir, step_index, x, y, &prog, current_color, next_rgb,
                        );
                        next[cell_idx] = Some((prog, next_rgb));
                    }
                    Err(err) => {
                        telemetry.run_failures += 1;
                        tracing::warn!(
                            x, y, step_index, error = %err,
                            "llm_cell_program_run_failed"
                        );
                    }
                },
                Err(err) => {
                    telemetry.llm_failures += 1;
                    tracing::warn!(
                        x, y, step_index, error = %err,
                        "llm_cell_propose_failed"
                    );
                }
            }
        }
    }

    for (cell_idx, slot) in next.into_iter().enumerate() {
        if let Some((prog, color)) = slot {
            cells[cell_idx].program = prog;
            cells[cell_idx].color = color;
        }
    }

    telemetry
}

/// Local-only fallback: just run each cell's existing program. No LLM.
fn local_evolve_step(
    cells: &mut [Cell],
    step_index: u32,
    (n, m): (usize, usize),
) -> StepTelemetry {
    let snapshot: Vec<Rgb> = cells.iter().map(|c| c.color).collect();
    let mut t = StepTelemetry::default();
    for x in 0..n {
        for y in 0..m {
            let neighbors: Vec<Rgb> = moore_neighbors((x, y), (n, m))
                .iter()
                .map(|&(nx, ny)| snapshot[index(nx, ny, m)])
                .collect();
            let cell = &mut cells[index(x, y, m)];
            match cell.program.run(cell.color, &neighbors, step_index) {
                Ok(next) => cell.color = next,
                Err(err) => {
                    t.run_failures += 1;
                    tracing::debug!(x, y, step_index, error = %err, "local_step_failed");
                }
            }
        }
    }
    t
}

fn build_per_cell_instruction(
    rule: &str,
    current_program: &CellProgram,
    current_color: Rgb,
    neighbors: &[Rgb],
    step_index: u32,
) -> String {
    let neighbors_repr = neighbors
        .iter()
        .map(|n| format!("({}, {}, {})", n.r, n.g, n.b))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Cell rule (keep consistent across ticks): {rule}\n\
         step_index: {step_index}\n\
         current_color (r,g,b): ({cr}, {cg}, {cb})\n\
         neighbor_colors: [{neighbors_repr}]\n\
         \n\
         Previous program:\n\
         ```python\n{prev_code}```\n\
         \n\
         Author the next CellProgram. You may keep `code` identical if it already produces interesting dynamics, or refine it. Return JSON only.",
        cr = current_color.r,
        cg = current_color.g,
        cb = current_color.b,
        prev_code = current_program.code,
    )
}

fn log_and_dump_program(
    dump_dir: Option<&Path>,
    step_index: u32,
    x: usize,
    y: usize,
    prog: &CellProgram,
    prev: Rgb,
    next: Rgb,
) {
    tracing::info!(
        x, y, step_index,
        rule = %prog.rule,
        prev_rgb = ?(prev.r, prev.g, prev.b),
        next_rgb = ?(next.r, next.g, next.b),
        code_preview = %preview_code(&prog.code),
        "llm_cell_evolved"
    );
    tracing::debug!(x, y, step_index, code = %prog.code, "llm_cell_program");

    if let Some(dir) = dump_dir {
        let step_dir = dir.join(format!("step_{step_index:04}"));
        if let Err(err) = std::fs::create_dir_all(&step_dir) {
            tracing::warn!(error = %err, dir = %step_dir.display(), "dump_dir_create_failed");
            return;
        }
        let path = step_dir.join(format!("cell_{x}_{y}.py"));
        let header = format!(
            "# step_index: {step_index}\n# cell: ({x}, {y})\n# rule: {rule}\n# prev_rgb: ({pr}, {pg}, {pb})\n# next_rgb: ({nr}, {ng}, {nb})\n\n",
            rule = prog.rule,
            pr = prev.r, pg = prev.g, pb = prev.b,
            nr = next.r, ng = next.g, nb = next.b,
        );
        if let Err(err) = std::fs::write(&path, format!("{header}{}", prog.code)) {
            tracing::warn!(error = %err, path = %path.display(), "dump_write_failed");
        }
    }
}

fn preview_code(code: &str) -> String {
    let one_line: String = code
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");
    if one_line.len() > 120 {
        format!("{}…", &one_line[..120])
    } else {
        one_line
    }
}

fn index(x: usize, y: usize, m: usize) -> usize {
    x * m + y
}

/// 8-cell Moore neighborhood with toroidal wrap.
fn moore_neighbors((x, y): (usize, usize), (n, m): (usize, usize)) -> [(usize, usize); 8] {
    let xm = (x + n - 1) % n;
    let xp = (x + 1) % n;
    let ym = (y + m - 1) % m;
    let yp = (y + 1) % m;
    [
        (xm, ym),
        (xm, y),
        (xm, yp),
        (x, ym),
        (x, yp),
        (xp, ym),
        (xp, y),
        (xp, yp),
    ]
}

/// A pleasant default gradient so the screen isn't blank before the first tick.
fn seed_color((x, y): (usize, usize), (n, m): (usize, usize)) -> Rgb {
    let r = ((x as f32 / n.max(1) as f32) * 255.0) as u8;
    let g = ((y as f32 / m.max(1) as f32) * 255.0) as u8;
    let b = 128;
    Rgb::new(r, g, b)
}

/// Local fallback program — small averaging rule with a slow warm drift.
fn demo_program() -> CellProgram {
    CellProgram {
        rule: "average neighbors with a slow warm drift".to_string(),
        code: r#"def step(state, neighbors, step_index):
    if not neighbors:
        return state
    n = len(neighbors)
    ar = sum(c[0] for c in neighbors) // n
    ag = sum(c[1] for c in neighbors) // n
    ab = sum(c[2] for c in neighbors) // n
    r = min(255, (state[0] + ar) // 2 + 1)
    g = max(0, (state[1] + ag) // 2)
    b = max(0, (state[2] + ab) // 2)
    return (r, g, b)
"#
        .to_string(),
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        })
        .unwrap_or(false)
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .pretty()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}
