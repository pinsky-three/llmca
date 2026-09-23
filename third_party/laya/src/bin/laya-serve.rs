//! A local web UI for the model: POST a state and typed questions, get calibrated answers
//! plus the exact token sequence the encoder saw.
//!
//! ```text
//! laya-serve --model models/laya-base --port 8080
//! ```

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use candle_core::Device;
use clap::Parser;
use laya::{Agent, Options, PromptView, Question, Response as LayaResponse};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const INDEX: &str = include_str!("../../web/index.html");
const LOGO: &[u8] = include_bytes!("../../web/logo.png");
const FAVICON: &[u8] = include_bytes!("../../web/favicon.png");

#[derive(Parser)]
#[command(about = "Local web UI for the Laya decision model", version)]
struct Args {
    #[arg(long, default_value = "models/laya-base")]
    model: PathBuf,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// cpu, metal, or cuda. Defaults to the best available.
    #[arg(long)]
    device: Option<String>,
}

struct AppState {
    agent: Agent,
    model_dir: String,
    device: String,
}

#[derive(Deserialize)]
struct PredictRequest {
    state: Value,
    /// Ordered, so the UI controls the display order rather than a map's iteration order.
    questions: Vec<NamedQuestion>,
    /// Also return the rendered token sequence per question.
    #[serde(default)]
    inspect: bool,
}

#[derive(Deserialize)]
struct NamedQuestion {
    id: String,
    #[serde(flatten)]
    question: Question,
}

#[derive(Serialize)]
struct PredictResponse {
    #[serde(flatten)]
    result: LayaResponse,
    /// Wall-clock time for the forward pass, in milliseconds.
    latency_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompts: Option<Vec<PromptView>>,
}

#[derive(Serialize)]
struct InfoResponse {
    model_dir: String,
    device: String,
    max_len: usize,
    head_max_len: usize,
}

/// Errors become a JSON body so the UI can show the model's own message (e.g. head_max_len).
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": format!("{:#}", self.0) });
        (StatusCode::BAD_REQUEST, Json(body)).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        AppError(e.into())
    }
}

async fn index() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        Html(INDEX),
    )
}

/// Static images, cached hard since they only change when the binary is rebuilt.
async fn png(bytes: &'static [u8]) -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "public, max-age=31536000, immutable"),
        ],
        bytes,
    )
}

async fn info(State(st): State<Arc<AppState>>) -> Json<InfoResponse> {
    Json(InfoResponse {
        model_dir: st.model_dir.clone(),
        device: st.device.clone(),
        max_len: st.agent.cfg.max_len,
        head_max_len: st.agent.cfg.head_max_len,
    })
}

async fn predict(
    State(st): State<Arc<AppState>>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, AppError> {
    // Inference is CPU/GPU-bound and synchronous, so keep it off the async runtime's threads.
    let out = tokio::task::spawn_blocking(move || -> Result<PredictResponse> {
        let questions: Vec<(String, Question)> = req
            .questions
            .into_iter()
            .map(|q| (q.id, q.question))
            .collect();

        let prompts = if req.inspect {
            let mut v = Vec::with_capacity(questions.len());
            for (id, q) in &questions {
                v.push(
                    st.agent
                        .render_prompt(&req.state, q)
                        .with_context(|| format!("question {id:?}"))?,
                );
            }
            Some(v)
        } else {
            None
        };

        let t0 = Instant::now();
        let result = st.agent.system_one(&req.state, &questions)?;
        Ok(PredictResponse {
            result,
            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
            prompts,
        })
    })
    .await??;
    Ok(Json(out))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let device = match args.device.as_deref() {
        None => None,
        Some("cpu") => Some(Device::Cpu),
        Some("metal") => Some(Device::new_metal(0).context("metal device unavailable")?),
        Some("cuda") => Some(Device::new_cuda(0).context("cuda device unavailable")?),
        Some(other) => anyhow::bail!("unknown device {other:?}"),
    };

    eprintln!("loading {} ...", args.model.display());
    let t0 = Instant::now();
    let agent = Agent::from_dir(
        &args.model,
        Options {
            device,
            dtype: None,
        },
    )?;
    let device = agent_device(&agent).to_string();
    eprintln!("loaded in {:.1}s on {device}", t0.elapsed().as_secs_f64());

    let st = Arc::new(AppState {
        model_dir: args.model.display().to_string(),
        device,
        agent,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/logo.png", get(|| png(LOGO)))
        .route("/favicon.png", get(|| png(FAVICON)))
        .route("/api/info", get(info))
        .route("/api/predict", post(predict))
        .with_state(st);

    let addr = format!("127.0.0.1:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("\n  http://{addr}\n");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn agent_device(agent: &Agent) -> &'static str {
    match agent.device() {
        Device::Cpu => "cpu",
        Device::Metal(_) => "metal",
        Device::Cuda(_) => "cuda",
    }
}
