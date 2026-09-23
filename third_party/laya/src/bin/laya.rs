//! CLI: answer typed questions about a state in one forward pass.
//!
//! ```text
//! laya --model models/laya-base --state-file ticket.txt --questions questions.json
//! ```

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use laya::{Agent, Options, Question};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    about = "Typed decisions from the Laya model, in one forward pass",
    version
)]
struct Args {
    /// Checkpoint directory (containing model.safetensors, encoder/, tokenizer/).
    #[arg(long, default_value = "models/laya-base")]
    model: PathBuf,

    /// The state to decide over, as literal text.
    #[arg(long, conflicts_with = "state_file")]
    state: Option<String>,

    /// The state to decide over, read from a file. Parsed as JSON if it parses, else used as text.
    #[arg(long)]
    state_file: Option<PathBuf>,

    /// Questions as a JSON object: {"id": {"type", "instructions", "criteria"}}.
    #[arg(long)]
    questions: PathBuf,

    /// cpu, metal, or cuda. Defaults to the best available.
    #[arg(long)]
    device: Option<String>,

    /// f16 or f32. Defaults to f32 on CPU, f16 on an accelerator.
    #[arg(long)]
    dtype: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let state: Value = match (&args.state, &args.state_file) {
        (Some(s), _) => Value::String(s.clone()),
        (None, Some(p)) => {
            let raw = std::fs::read_to_string(p)
                .with_context(|| format!("reading state {}", p.display()))?;
            serde_json::from_str(&raw).unwrap_or(Value::String(raw))
        }
        (None, None) => anyhow::bail!("pass --state or --state-file"),
    };

    let raw = std::fs::read_to_string(&args.questions)
        .with_context(|| format!("reading questions {}", args.questions.display()))?;
    // BTreeMap keeps the output order stable; each question's own options keep their JSON order.
    let questions: BTreeMap<String, Question> =
        serde_json::from_str(&raw).context("parsing questions JSON")?;
    let questions: Vec<(String, Question)> = questions.into_iter().collect();

    let opts = Options {
        device: match args.device.as_deref() {
            None => None,
            Some("cpu") => Some(Device::Cpu),
            Some("metal") => Some(Device::new_metal(0).context("metal device unavailable")?),
            Some("cuda") => Some(Device::new_cuda(0).context("cuda device unavailable")?),
            Some(other) => anyhow::bail!("unknown device {other:?}"),
        },
        dtype: match args.dtype.as_deref() {
            None => None,
            Some("f16") => Some(DType::F16),
            Some("f32") => Some(DType::F32),
            Some(other) => anyhow::bail!("unknown dtype {other:?}"),
        },
    };

    let agent = Agent::from_dir(&args.model, opts)?;
    let response = agent.system_one(&state, &questions)?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}
