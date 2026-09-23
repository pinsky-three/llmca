use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use laya_ca::{
    centered_blinker, colors, evolve, life_describe, life_options, life_step, probes, score, soup,
    swatch_for_label, with_center, Decider, Lattice, LifeScore, MajorityDecider, Prompt,
    StepConfig, StepReport, LIFE_INSTRUCTIONS,
};
use macroquad::prelude::*;
use tracing_subscriber::EnvFilter;

#[cfg(feature = "infer")]
use laya_ca::{DecideError, LayaDecider};

/// Fastest the picture is allowed to advance. Inference is usually slower.
const MIN_STEP: Duration = Duration::from_millis(200);

fn window_conf() -> Conf {
    Conf {
        window_title: "LLMCA Laya".to_owned(),
        window_width: 800,
        window_height: 800,
        ..Default::default()
    }
}

fn main() {
    init_tracing();
    let args = Args::parse();
    if let Err(err) = validate(&args) {
        tracing::error!(error = %err, "laya_ca_bad_args");
        eprintln!("{err}");
        std::process::exit(2);
    }
    if args.spot {
        if let Err(err) = run_spot(&args) {
            tracing::error!(error = %err, "laya_ca_life_failed");
            eprintln!("{err}");
            std::process::exit(1);
        }
        return;
    }
    if args.probe {
        if let Err(err) = run_life(&args) {
            tracing::error!(error = %err, "laya_ca_life_failed");
            eprintln!("{err}");
            std::process::exit(1);
        }
        return;
    }
    let life = args.life;
    let temperature = if life {
        args.temperature.unwrap_or(1.0)
    } else {
        args.temperature.unwrap_or(4.0)
    };
    let conway = args.conway;
    let mut conf = window_conf();
    if life {
        conf.window_title = "Laya Game of Life".to_owned();
    } else if conway {
        conf.window_title = "Game of Life".to_owned();
    }
    macroquad::Window::from_config(conf, viewer(args, temperature, conway, life));
}
#[derive(Debug, Parser, Clone)]
#[command(
    name = "laya-ca",
    about = "Cellular automaton of Laya typed color decisions"
)]
struct Args {
    /// Cells on each axis. Each cell is one ModernBERT forward pass.
    #[arg(long, default_value_t = 50)]
    width: usize,
    #[arg(long, default_value_t = 50)]
    height: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Keep the current color when the decision is less confident than this.
    #[arg(long, default_value_t = 0.0)]
    min_confidence: f32,
    /// `1` keeps the argmax. Higher values flatten the color distribution, then sample.
    /// Omitted: 4 for the viewer, 1 for `--life`.
    #[arg(long)]
    temperature: Option<f32>,
    /// Cells scored together in one forward pass. `1` is the fastest on this GPU.
    #[arg(long, default_value_t = 1)]
    batch: usize,
    /// Run the encoder in f16. The sunset picture does this unless you are in `--life`.
    #[arg(long, default_value_t = false)]
    f16: bool,
    /// Checkpoint directory. Also read from LAYA_MODEL_DIR.
    #[arg(long, env = "LAYA_MODEL_DIR")]
    model: Option<String>,
    /// Majority-of-neighbors colors, with no checkpoint.
    #[arg(long, default_value_t = false)]
    fake: bool,
    /// Laya chooses each next color under Conway's rule, shown in the window.
    #[arg(long, default_value_t = false)]
    life: bool,
    /// Start `--life` from a random binary field instead of a blinker.
    #[arg(long, default_value_t = false)]
    soup: bool,
    /// Headless score of a blinker, a block, and a glider, then exit.
    #[arg(long, default_value_t = false)]
    probe: bool,
    /// Ask Laya the 18 Life situations once, then exit.
    #[arg(long, default_value_t = false)]
    spot: bool,
    /// Exact Conway rule in the window. No checkpoint.
    #[arg(long, default_value_t = false)]
    conway: bool,
}

enum Event {
    Loading,
    Frame {
        lattice: Lattice,
        report: StepReport,
        step_idx: u64,
        life: Option<LifeScore>,
    },
    Failed(String),
}

async fn viewer(args: Args, temperature: f32, conway: bool, life: bool) {
    let (tx, rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        if args.life {
            laya_life_worker(args, tx);
        } else if args.conway {
            conway_worker(args, tx);
        } else {
            worker(args, tx);
        }
    });

    // The checkpoint load stays on the worker so this loop can paint.
    let mut latest = Event::Loading;

    loop {
        while let Ok(event) = rx.try_recv() {
            latest = event;
        }
        clear_background(BLACK);
        match &latest {
            Event::Loading => {
                draw_outlined_text("loading checkpoint", 24.0, 40.0, 28.0);
            }
            Event::Failed(message) => {
                draw_outlined_text(message, 24.0, 40.0, 24.0);
            }
            Event::Frame {
                lattice,
                report,
                step_idx,
                life: life_score,
            } => {
                draw_lattice(lattice);
                let status = if let Some(scored) = life_score {
                    format!(
                        "laya life {step_idx}  {}x{}  changes {}/{}  stays {}/{}  exact {}  conf {:.2}  {} ms",
                        lattice.width(),
                        lattice.height(),
                        scored.changed_correct,
                        scored.should_change,
                        scored.stayed_correct,
                        scored.should_stay,
                        if scored.exact { "yes" } else { "no" },
                        report.mean_confidence,
                        report.elapsed_ms
                    )
                } else if conway || life {
                    format!(
                        "life {step_idx}  {}x{}  {} live  {} changed",
                        lattice.width(),
                        lattice.height(),
                        live_cells(lattice),
                        report.rewrites
                    )
                } else {
                    format!(
                        "step {step_idx}  {}x{}  {} colors  conf {:.2}  tau {:.0}  {} ms  {} fail",
                        lattice.width(),
                        lattice.height(),
                        report.unique_swatches,
                        report.mean_confidence,
                        temperature,
                        report.elapsed_ms,
                        report.failures
                    )
                };
                draw_outlined_text(&status, 16.0, screen_height() - 20.0, 22.0);
            }
        }
        next_frame().await;
    }
}

fn laya_life_worker(args: Args, tx: mpsc::SyncSender<Event>) {
    let decider = match build_decider(&args) {
        Ok(decider) => decider,
        Err(err) => {
            tracing::error!(error = %err, "laya_ca_decider_unavailable");
            let _ = tx.send(Event::Failed(err));
            return;
        }
    };
    let lattice = if args.soup {
        soup(args.width, args.height, args.seed)
    } else {
        centered_blinker(args.width, args.height)
    };
    let mut lattice = match lattice {
        Ok(lattice) => lattice,
        Err(err) => {
            let _ = tx.send(Event::Failed(err.to_string()));
            return;
        }
    };
    let cfg = StepConfig {
        min_confidence: args.min_confidence,
        temperature: args.temperature.unwrap_or(1.0),
        instructions: LIFE_INSTRUCTIONS.to_string(),
        prompt: Prompt::Life,
        batch: args.batch,
    };
    let mut rng = args.seed | 1;
    tracing::info!(
        width = lattice.width(),
        height = lattice.height(),
        temperature = cfg.temperature,
        soup = args.soup,
        seed = args.seed,
        "laya_ca_life"
    );
    if tx
        .send(Event::Frame {
            lattice: lattice.clone(),
            report: StepReport {
                cells: lattice.width() * lattice.height(),
                unique_swatches: lattice.unique_swatches(),
                ..StepReport::default()
            },
            step_idx: 0,
            life: None,
        })
        .is_err()
    {
        return;
    }

    let mut step_idx = 0u64;
    loop {
        let started = Instant::now();
        let (next, report) = evolve(&lattice, decider.as_ref(), &cfg, &mut rng);
        let scored = score(&lattice, &next);
        tracing::info!(
            generation = step_idx + 1,
            exact = scored.exact,
            changed = scored.changed_correct,
            should_change = scored.should_change,
            stayed = scored.stayed_correct,
            should_stay = scored.should_stay,
            "laya_ca_life_score"
        );
        lattice = next;
        step_idx += 1;
        if tx
            .send(Event::Frame {
                lattice: lattice.clone(),
                report,
                step_idx,
                life: Some(scored),
            })
            .is_err()
        {
            break;
        }
        thread::sleep(MIN_STEP.saturating_sub(started.elapsed()));
    }
}

fn conway_worker(args: Args, tx: mpsc::SyncSender<Event>) {
    let mut lattice = match soup(args.width, args.height, args.seed) {
        Ok(lattice) => lattice,
        Err(err) => {
            let _ = tx.send(Event::Failed(err.to_string()));
            return;
        }
    };
    tracing::info!(
        width = lattice.width(),
        height = lattice.height(),
        seed = args.seed,
        "laya_ca_conway"
    );
    if tx
        .send(Event::Frame {
            lattice: lattice.clone(),
            report: StepReport {
                cells: lattice.width() * lattice.height(),
                unique_swatches: lattice.unique_swatches(),
                ..StepReport::default()
            },
            step_idx: 0,
            life: None,
        })
        .is_err()
    {
        return;
    }

    let mut step_idx = 0u64;
    loop {
        let started = Instant::now();
        let next = life_step(&lattice);
        let rewrites = changed_cells(&lattice, &next);
        let report = StepReport {
            cells: next.width() * next.height(),
            rewrites,
            unchanged: next.width() * next.height() - rewrites,
            unique_swatches: next.unique_swatches(),
            elapsed_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
            ..StepReport::default()
        };
        lattice = next;
        step_idx += 1;
        if tx
            .send(Event::Frame {
                lattice: lattice.clone(),
                report,
                step_idx,
                life: None,
            })
            .is_err()
        {
            break;
        }
        thread::sleep(Duration::from_millis(80).saturating_sub(started.elapsed()));
    }
}

fn changed_cells(before: &Lattice, after: &Lattice) -> usize {
    let mut changed = 0;
    for y in 0..before.height() {
        for x in 0..before.width() {
            if before.cell(x, y).swatch != after.cell(x, y).swatch {
                changed += 1;
            }
        }
    }
    changed
}

fn live_cells(lattice: &Lattice) -> usize {
    let mut live = 0;
    for y in 0..lattice.height() {
        for x in 0..lattice.width() {
            if palette_rgba(lattice.cell(x, y).swatch) == [255, 255, 255, 255] {
                live += 1;
            }
        }
    }
    live
}

fn worker(args: Args, tx: mpsc::SyncSender<Event>) {
    let decider = match build_decider(&args) {
        Ok(decider) => decider,
        Err(err) => {
            tracing::error!(error = %err, "laya_ca_decider_unavailable");
            let _ = tx.send(Event::Failed(err));
            return;
        }
    };
    let mut lattice = match Lattice::seeded(args.width, args.height, args.seed) {
        Ok(lattice) => lattice,
        Err(err) => {
            let _ = tx.send(Event::Failed(err.to_string()));
            return;
        }
    };
    let cfg = StepConfig {
        min_confidence: args.min_confidence,
        temperature: args.temperature.unwrap_or(4.0),
        batch: args.batch,
        ..StepConfig::default()
    };
    let mut rng = args.seed | 1;
    let opening = StepReport {
        cells: lattice.width() * lattice.height(),
        unique_swatches: lattice.unique_swatches(),
        ..StepReport::default()
    };
    if tx
        .send(Event::Frame {
            lattice: lattice.clone(),
            report: opening,
            step_idx: 0,
            life: None,
        })
        .is_err()
    {
        return;
    }

    let mut step_idx = 0u64;
    loop {
        let started = Instant::now();
        let (next, report) = evolve(&lattice, decider.as_ref(), &cfg, &mut rng);
        lattice = next;
        step_idx += 1;
        if tx
            .send(Event::Frame {
                lattice: lattice.clone(),
                report,
                step_idx,
                life: None,
            })
            .is_err()
        {
            break;
        }
        // A local decider finishes in microseconds. The floor keeps a settled
        // picture from pegging a core and flooding the step log.
        thread::sleep(MIN_STEP.saturating_sub(started.elapsed()));
    }
}

fn build_decider(args: &Args) -> Result<Box<dyn Decider>, String> {
    if args.fake {
        tracing::info!("laya_ca_majority_decider");
        return Ok(Box::new(MajorityDecider));
    }
    open_laya(args)
}

#[cfg(feature = "infer")]
fn open_laya(args: &Args) -> Result<Box<dyn Decider>, String> {
    let Some(dir) = args.model.as_deref() else {
        return Err(
            "pass --model or set LAYA_MODEL_DIR to a Laya checkpoint (or pass --fake)".into(),
        );
    };
    // Sunset's cost is the encoder. f16 is the fast Metal path there.
    // Life stays f32 unless --f16 is set: the birth call is a near tie.
    let half = args.f16 || !(args.life || args.spot || args.probe);
    LayaDecider::open(std::path::Path::new(dir), half)
        .map(|decider| Box::new(decider) as Box<dyn Decider>)
        .map_err(|err: DecideError| err.to_string())
}

#[cfg(not(feature = "infer"))]
fn open_laya(args: &Args) -> Result<Box<dyn Decider>, String> {
    let _ = args;
    Err(
        "this build has no Laya runtime. Re-run with --features infer (and metal on Apple GPU), or pass --fake."
            .into(),
    )
}

fn run_spot(args: &Args) -> Result<(), String> {
    let decider = build_decider(args)?;
    let options = life_options();
    let mut hits = 0u32;
    let mut asked = 0u32;
    for occupied in [false, true] {
        for neighbors in 0..=8 {
            let lattice = with_center(occupied, neighbors);
            let truth = life_step(&lattice);
            let want = if truth.cell(1, 1).swatch == swatch_for_label("on").unwrap() {
                "on"
            } else {
                "off"
            };
            let state = life_describe(&lattice, 1, 1);
            let choice = decider
                .choose(&state, LIFE_INSTRUCTIONS, &options)
                .map_err(|err| err.to_string())?;
            let got = choice.label.trim().to_ascii_lowercase();
            let ok = got == want;
            if ok {
                hits += 1;
            }
            asked += 1;
            println!(
                "{state}  got {got}  want {want}  {}  conf {:.3}  probs {}",
                if ok { "ok" } else { "MISS" },
                choice.confidence,
                choice
                    .probabilities
                    .iter()
                    .map(|(label, p)| format!("{label}={p:.3}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    println!("spot {hits}/{asked}");
    Ok(())
}

fn run_life(args: &Args) -> Result<(), String> {
    let decider = build_decider(args)?;
    let temperature = args.temperature.unwrap_or(1.0);
    let cfg = StepConfig {
        min_confidence: args.min_confidence,
        temperature,
        instructions: LIFE_INSTRUCTIONS.to_string(),
        prompt: Prompt::Life,
        batch: args.batch,
    };
    let mut rng = args.seed | 1;
    let mut seen = 0u32;
    let mut exact = 0u32;
    println!("life probe  temperature {temperature}");
    for probe in probes() {
        let mut lattice = probe.lattice;
        for generation in 1..=probe.generations {
            let (next, report) = evolve(&lattice, decider.as_ref(), &cfg, &mut rng);
            let scored = score(&lattice, &next);
            seen += 1;
            if scored.exact {
                exact += 1;
            }
            println!(
                "{name} gen {generation}  exact {exact_word}  changes {changed}/{should_change}  stays {stayed}/{should_stay}  match {matches}/{cells}  conf {conf:.2}  {ms} ms  fail {failures}",
                name = probe.name,
                exact_word = if scored.exact { "yes" } else { "no" },
                changed = scored.changed_correct,
                should_change = scored.should_change,
                stayed = scored.stayed_correct,
                should_stay = scored.should_stay,
                matches = scored.matches,
                cells = scored.cells,
                conf = report.mean_confidence,
                ms = report.elapsed_ms,
                failures = report.failures,
            );
            lattice = next;
        }
    }
    println!("life exact {exact}/{seen}");
    Ok(())
}

fn validate(args: &Args) -> Result<(), String> {
    if args.batch == 0 {
        return Err("batch must be at least 1".into());
    }
    if let Some(temperature) = args.temperature {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(format!("temperature must be positive, got {temperature}"));
        }
    }
    if !(0.0..=1.0).contains(&args.min_confidence) {
        return Err(format!(
            "min-confidence must be between 0 and 1, got {}",
            args.min_confidence
        ));
    }
    Lattice::seeded(args.width, args.height, args.seed)
        .map(|_| ())
        .map_err(|err| err.to_string())
}

/// White glyphs with a black ring so the line stays readable on any cell color.
fn draw_outlined_text(text: &str, x: f32, y: f32, size: f32) {
    let ink = Color::from_rgba(0, 0, 0, 255);
    for dy in -2..=2 {
        for dx in -2..=2 {
            if dx == 0 && dy == 0 {
                continue;
            }
            draw_text(text, x + dx as f32, y + dy as f32, size, ink);
        }
    }
    draw_text(text, x, y, size, WHITE);
}

fn draw_lattice(lattice: &Lattice) {
    let cell =
        (screen_width() / lattice.width() as f32).min(screen_height() / lattice.height() as f32);
    for y in 0..lattice.height() {
        for x in 0..lattice.width() {
            let [r, g, b, a] = palette_rgba(lattice.cell(x, y).swatch);
            draw_rectangle(
                x as f32 * cell,
                y as f32 * cell,
                cell,
                cell,
                Color::from_rgba(r, g, b, a),
            );
        }
    }
}

fn palette_rgba(index: usize) -> [u8; 4] {
    colors()
        .get(index)
        .map(|swatch| swatch.rgba)
        .unwrap_or([0, 0, 0, 255])
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .json()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}
