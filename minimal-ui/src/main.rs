//! Minimal UI for the programmable-cell PoC.
//!
//! Architecture (shared-program CA):
//!
//!   - One `CellProgram` drives EVERY cell. Real CA dynamics emerge from
//!     applying one rule across many cells over many ticks.
//!   - The compute loop ticks the program locally (monty, microseconds) at
//!     `LLMCA_COMPUTE_HZ` (default 8 Hz) — that's where the dynamics live.
//!   - The render loop runs at 60 Hz and lerps each cell's color from its
//!     previous to its target value, so transitions are smooth gradients.
//!   - The LLM is consulted in a background tokio task at epoch boundaries
//!     (`LLMCA_TICKS_PER_EPOCH`, default 50). It rewrites/refines the rule
//!     based on a world-state summary; new programs hot-swap atomically.

use dotenv::dotenv;
use dynamical_system::system::{
    programmable::{propose_program_via_resolver, CellProgram, Rgb},
    space::{load_llm_resolvers, LLMResolver},
};
use itertools::Itertools;
use macroquad::prelude::*;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;

const DEFAULT_GRID_SIZE: usize = 64;
const DEFAULT_RULE: &str =
    "Pick a state-of-the-art continuous CA family (Lenia, Gray-Scott RD, cyclic CA, or Brian's Brain) that yields visibly moving patterns: gliders, waves, spirals or mitosis.";
const PROGRAMS_DIR: &str = ".minimal_ui_programs";
const DEFAULT_FADE_MS: u64 = 200;
const DEFAULT_COMPUTE_HZ: u64 = 8;
const DEFAULT_TICKS_PER_EPOCH: u32 = 50;

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
    prev_color: Rgb,
    target_color: Rgb,
    transition_started_at: Instant,
}

impl Cell {
    fn new(initial: Rgb) -> Self {
        Self {
            prev_color: initial,
            target_color: initial,
            transition_started_at: Instant::now(),
        }
    }

    fn rendered_color(&self, now: Instant, fade: Duration) -> Rgb {
        if fade.is_zero() {
            return self.target_color;
        }
        let elapsed = now.saturating_duration_since(self.transition_started_at);
        let t = (elapsed.as_secs_f32() / fade.as_secs_f32()).clamp(0.0, 1.0);
        lerp_rgb(self.prev_color, self.target_color, ease_in_out(t))
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    dotenv().ok();
    init_tracing();
    let startup_started_at = Instant::now();

    let rt = tokio::runtime::Runtime::new().unwrap();
    let _enter = rt.enter();

    let grid = env_usize("LLMCA_GRID").unwrap_or(DEFAULT_GRID_SIZE).max(2);
    let (n, m) = (grid, grid);
    let rule = std::env::var("LLMCA_RULE").unwrap_or_else(|_| DEFAULT_RULE.to_string());
    let fade = Duration::from_millis(
        env_usize("LLMCA_FADE_MS")
            .map(|v| v as u64)
            .unwrap_or(DEFAULT_FADE_MS),
    );
    let compute_hz = env_usize("LLMCA_COMPUTE_HZ")
        .map(|v| v as u64)
        .unwrap_or(DEFAULT_COMPUTE_HZ)
        .max(1);
    let compute_period = Duration::from_millis(1000 / compute_hz);
    let ticks_per_epoch = env_usize("LLMCA_TICKS_PER_EPOCH")
        .map(|v| v as u32)
        .unwrap_or(DEFAULT_TICKS_PER_EPOCH);

    let llm_disabled = env_flag("LLMCA_NO_LLM");
    let resolvers: Vec<LLMResolver> = if llm_disabled {
        tracing::info!("minimal_ui_llm_disabled_using_local_demo_program");
        vec![]
    } else {
        let r = load_llm_resolvers();
        if r.is_empty() {
            tracing::warn!(
                "no resolvers configured — populate ./resolvers.toml or set OPENAI_API_URL/MODEL_NAME/API_KEY. Falling back to local demo."
            );
        }
        r
    };

    let dump_dir = PathBuf::from(PROGRAMS_DIR);
    let _ = std::fs::create_dir_all(&dump_dir);

    let mut cells: Vec<Cell> = (0..n)
        .cartesian_product(0..m)
        .map(|(x, y)| Cell::new(seed_color((x, y), (n, m))))
        .collect();

    // Start with a known-good cyclic CA so the screen moves while the first
    // LLM call is in flight. The LLM may replace it on the first epoch.
    let mut program = if resolvers.is_empty() {
        demo_program()
    } else {
        seed_program()
    };
    let mut runner = program.compile().unwrap_or_else(|err| {
        tracing::error!(error = %err, "seed_program_failed_to_compile");
        std::process::exit(1);
    });

    tracing::info!(
        n,
        m,
        resolvers = resolvers.len(),
        dump_dir = %dump_dir.display(),
        compute_hz,
        ticks_per_epoch,
        fade_ms = fade.as_millis() as u64,
        rule = %rule,
        temperature = std::env::var("LLMCA_TEMP").unwrap_or_else(|_| "0.7 (default)".into()),
        startup_elapsed_ms = startup_started_at.elapsed().as_millis() as u64,
        "minimal_ui_ready"
    );

    let mut compute_step: u32 = 0;
    let mut epoch: u32 = 0;
    let mut last_compute_at = Instant::now();

    let (program_tx, mut program_rx) = mpsc::unbounded_channel::<(u32, CellProgram)>();
    let mut llm_in_flight = false;

    // Kick off the first authoring immediately if we have resolvers.
    if !resolvers.is_empty() {
        spawn_llm_authoring(
            &rt,
            resolvers.clone(),
            rule.clone(),
            None,
            world_state_summary(&cells, n, m),
            epoch,
            program_tx.clone(),
        );
        llm_in_flight = true;
    }

    loop {
        let now = Instant::now();

        // 1. Drain LLM channel; hot-swap shared program if a new one arrived.
        while let Ok((received_epoch, new_prog)) = program_rx.try_recv() {
            match new_prog.compile() {
                Ok(new_runner) => {
                    log_and_dump_program(Some(&dump_dir), received_epoch, &new_prog);
                    program = new_prog;
                    runner = new_runner;
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        epoch = received_epoch,
                        "llm_program_compile_failed_keeping_previous"
                    );
                }
            }
            llm_in_flight = false;
        }

        // 2. Render interpolated colors.
        let cell_size = (screen_width() / n as f32).min(screen_height() / m as f32);
        for x in 0..n {
            for y in 0..m {
                let c = cells[index(x, y, m)].rendered_color(now, fade);
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

        // 3. Compute step at LLMCA_COMPUTE_HZ.
        if now.saturating_duration_since(last_compute_at) >= compute_period {
            let snapshot: Vec<Rgb> = cells.iter().map(|c| c.target_color).collect();
            let mut run_failures = 0_usize;
            for x in 0..n {
                for y in 0..m {
                    let neighbors: Vec<Rgb> = moore_neighbors((x, y), (n, m))
                        .iter()
                        .map(|&(nx, ny)| snapshot[index(nx, ny, m)])
                        .collect();
                    let cell = &mut cells[index(x, y, m)];
                    match CellProgram::run_compiled(
                        &runner,
                        cell.target_color,
                        &neighbors,
                        compute_step,
                    ) {
                        Ok(next) => {
                            cell.prev_color = cell.rendered_color(now, fade);
                            cell.target_color = next;
                            cell.transition_started_at = now;
                        }
                        Err(_) => run_failures += 1,
                    }
                }
            }
            if run_failures > 0 {
                tracing::debug!(
                    compute_step,
                    run_failures,
                    "minimal_ui_compute_step_partial_failure"
                );
            }
            last_compute_at = now;
            compute_step += 1;

            // 4. Trigger the next LLM authoring at epoch boundaries.
            if !resolvers.is_empty()
                && !llm_in_flight
                && ticks_per_epoch > 0
                && compute_step % ticks_per_epoch == 0
            {
                epoch += 1;
                tracing::info!(
                    epoch,
                    compute_step,
                    "minimal_ui_epoch_boundary_requesting_new_program"
                );
                spawn_llm_authoring(
                    &rt,
                    resolvers.clone(),
                    rule.clone(),
                    Some(program.clone()),
                    world_state_summary(&cells, n, m),
                    epoch,
                    program_tx.clone(),
                );
                llm_in_flight = true;
            }
        }

        // 5. R resets the world (re-seed with a known-good rule and reroll).
        if is_key_pressed(KeyCode::R) {
            tracing::info!("minimal_ui_reset_requested");
            for (i, (x, y)) in (0..n).cartesian_product(0..m).enumerate() {
                cells[i] = Cell::new(seed_color((x, y), (n, m)));
            }
            program = if resolvers.is_empty() {
                demo_program()
            } else {
                seed_program()
            };
            if let Ok(r) = program.compile() {
                runner = r;
            }
            if !resolvers.is_empty() && !llm_in_flight {
                epoch += 1;
                spawn_llm_authoring(
                    &rt,
                    resolvers.clone(),
                    rule.clone(),
                    None,
                    world_state_summary(&cells, n, m),
                    epoch,
                    program_tx.clone(),
                );
                llm_in_flight = true;
            }
        }

        next_frame().await;
    }
}

fn spawn_llm_authoring(
    rt: &tokio::runtime::Runtime,
    resolvers: Vec<LLMResolver>,
    rule: String,
    previous: Option<CellProgram>,
    world: WorldSummary,
    epoch: u32,
    tx: mpsc::UnboundedSender<(u32, CellProgram)>,
) {
    let resolver = resolvers[0].clone(); // shared program → one resolver is enough
    let started_at = Instant::now();

    rt.spawn(async move {
        let instruction = build_instruction(&rule, previous.as_ref(), &world, epoch);
        match propose_program_via_resolver(&resolver, &instruction).await {
            Ok(prog) => {
                tracing::info!(
                    epoch,
                    rule = %prog.rule,
                    code_chars = prog.code.len(),
                    elapsed_ms = started_at.elapsed().as_millis() as u64,
                    "llm_program_received"
                );
                let _ = tx.send((epoch, prog));
            }
            Err(err) => {
                tracing::warn!(
                    epoch,
                    error = %err,
                    elapsed_ms = started_at.elapsed().as_millis() as u64,
                    "llm_program_request_failed_keeping_previous"
                );
            }
        }
    });
}

#[derive(Debug, Clone, Copy)]
struct WorldSummary {
    avg: (u8, u8, u8),
    std: (u8, u8, u8),
    distinct_buckets: usize,
    grid: (usize, usize),
}

fn world_state_summary(cells: &[Cell], n: usize, m: usize) -> WorldSummary {
    let total = cells.len() as f32;
    let (mut sr, mut sg, mut sb) = (0.0_f32, 0.0_f32, 0.0_f32);
    for c in cells {
        sr += c.target_color.r as f32;
        sg += c.target_color.g as f32;
        sb += c.target_color.b as f32;
    }
    let (mr, mg, mb) = (sr / total, sg / total, sb / total);
    let (mut vr, mut vg, mut vb) = (0.0_f32, 0.0_f32, 0.0_f32);
    for c in cells {
        vr += (c.target_color.r as f32 - mr).powi(2);
        vg += (c.target_color.g as f32 - mg).powi(2);
        vb += (c.target_color.b as f32 - mb).powi(2);
    }
    let std_r = (vr / total).sqrt() as u8;
    let std_g = (vg / total).sqrt() as u8;
    let std_b = (vb / total).sqrt() as u8;

    let bucket =
        |c: Rgb| -> u32 { ((c.r as u32 / 32) << 6) | ((c.g as u32 / 32) << 3) | (c.b as u32 / 32) };
    let mut buckets = std::collections::HashSet::new();
    for c in cells {
        buckets.insert(bucket(c.target_color));
    }

    WorldSummary {
        avg: (mr.round() as u8, mg.round() as u8, mb.round() as u8),
        std: (std_r, std_g, std_b),
        distinct_buckets: buckets.len(),
        grid: (n, m),
    }
}

fn build_instruction(
    rule: &str,
    previous: Option<&CellProgram>,
    world: &WorldSummary,
    epoch: u32,
) -> String {
    let (n, m) = world.grid;
    let prev_block = match previous {
        Some(p) => format!(
            "Previous program (rule: {prev_rule}):\n```python\n{code}```\n",
            prev_rule = p.rule,
            code = p.code,
        ),
        None => "Previous program: <none — author the first one>\n".to_string(),
    };
    let avg = world.avg;
    let std = world.std;
    format!(
        "Goal rule: {rule}\n\
         epoch: {epoch}\n\
         grid: {n}x{m} toroidal\n\
         world_state: avg_rgb=({ar},{ag},{ab}) std_rgb=({sr},{sg},{sb}) distinct_color_buckets={buckets}/512\n\
         \n\
         {prev_block}\
         \n\
         If the previous program already produced visible motion (high std or growing distinct_buckets),\n\
         keep its archetype and refine parameters. If the world looks flat (low std, few buckets), SWITCH\n\
         archetype — try Lenia, Gray-Scott, cyclic CA or Brian's Brain (see system prompt examples).\n\
         Return JSON only.",
        ar = avg.0, ag = avg.1, ab = avg.2,
        sr = std.0, sg = std.1, sb = std.2,
        buckets = world.distinct_buckets,
    )
}

fn log_and_dump_program(dump_dir: Option<&Path>, epoch: u32, prog: &CellProgram) {
    tracing::info!(
        epoch,
        rule = %prog.rule,
        code_preview = %preview_code(&prog.code),
        "llm_program_active"
    );
    tracing::debug!(epoch, code = %prog.code, "llm_program_code");

    if let Some(dir) = dump_dir {
        let path = dir.join(format!("epoch_{epoch:04}.py"));
        let header = format!("# epoch: {epoch}\n# rule: {}\n\n", prog.rule);
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
    if one_line.len() > 140 {
        format!("{}…", &one_line[..140])
    } else {
        one_line
    }
}

fn index(x: usize, y: usize, m: usize) -> usize {
    x * m + y
}

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

/// Initial seed pattern: a few coloured "spikes" on a low-noise background.
/// Helps Lenia/Gray-Scott actually start (a flat field is a fixed point).
fn seed_color((x, y): (usize, usize), (n, m): (usize, usize)) -> Rgb {
    let cx = n as i32 / 2;
    let cy = m as i32 / 2;
    let dx = x as i32 - cx;
    let dy = y as i32 - cy;
    let r2 = (dx * dx + dy * dy) as f32;
    let radius_sq = (n.min(m) as f32 / 4.0).powi(2);
    let intensity = (-r2 / radius_sq).exp();
    let r = (intensity * 230.0 + 25.0) as u8;
    let g = (((x ^ y) & 0x1f) as u8).saturating_mul(8);
    let b = ((1.0 - intensity) * 200.0 + 30.0) as u8;
    Rgb::new(r, g, b)
}

fn lerp_rgb(a: Rgb, b: Rgb, t: f32) -> Rgb {
    let mix = |x: u8, y: u8| -> u8 {
        let v = x as f32 + (y as f32 - x as f32) * t;
        v.round().clamp(0.0, 255.0) as u8
    };
    Rgb::new(mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b))
}

fn ease_in_out(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Fallback when no LLM is configured — Lenia growth on the red channel.
/// Same Python the system prompt cites as Example 2; produces visible
/// breathing/glider patterns from the seed.
fn demo_program() -> CellProgram {
    CellProgram {
        rule: "Lenia growth on red channel (mu=0.35, sigma=0.07)".to_string(),
        code: r#"def step(state, neighbors, step_index):
    if not neighbors:
        return state
    K = sum(c[0] for c in neighbors) / (255.0 * len(neighbors))
    mu, sigma, dt = 0.35, 0.07, 0.15
    G = 2.0 * (2.71828 ** (-((K - mu) ** 2) / (2 * sigma * sigma))) - 1.0
    u = state[0] / 255.0
    new_u = max(0.0, min(1.0, u + dt * G))
    r = int(new_u * 255)
    g = int((new_u * 0.6 + 0.2) * 255)
    b = int((1.0 - new_u) * 255)
    return (r, g, b)
"#
        .to_string(),
    }
}

/// Initial program used while the first LLM call is in flight. Cyclic CA —
/// produces immediately visible spiral waves so the user has motion within
/// the first frame instead of staring at a static gradient.
fn seed_program() -> CellProgram {
    CellProgram {
        rule: "cyclic CA, N=12 (seed program before first LLM call)".to_string(),
        code: r#"def step(state, neighbors, step_index):
    N = 12
    s = (state[0] * N) // 256
    succ = (s + 1) % N
    next_s = s
    for c in neighbors:
        ns = (c[0] * N) // 256
        if ns == succ:
            next_s = succ
            break
    v = (next_s * 255) // (N - 1)
    return (v, (v + 80) % 256, (255 - v))
"#
        .to_string(),
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
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
