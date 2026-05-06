//! Programmable cells: each cell carries a tiny Python `step` function written
//! by an LLM. The function is executed every tick by `monty` (a sandboxed Python
//! interpreter) and returns the cell's next RGB color.
//!
//! Why: we want to test the hypothesis that LLMs can author *real* dynamical
//! systems — not just guess the next hex string. Pushing the contract down to
//! "produce a typed Python function" lets us measure expressiveness, stability
//! and correctness with concrete, executable artifacts.

use monty::{MontyObject, MontyRun, NoLimitTracker, PrintWriter};
use reqwest::Client as HttpClient;
use rig::{
    client::{CompletionClient, Nothing},
    completion::TypedPrompt,
    providers::ollama,
};
use serde_json::{json, Value};
use schemars::{schema_for, JsonSchema};
use serde_derive::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::system::space::LLMResolver;
use crate::system::unit::{CognitiveContext, LLMProvider};

/// Standard 8-bit RGB color used both as cell state and render value.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub const fn black() -> Self {
        Self::new(0, 0, 0)
    }

    pub fn from_hex(hex: &str) -> Option<Self> {
        let trimmed = hex.trim_matches(['#', '"', '[', ']']).to_lowercase();
        if trimmed.len() < 6 {
            return None;
        }
        let r = u8::from_str_radix(&trimmed[0..2], 16).ok()?;
        let g = u8::from_str_radix(&trimmed[2..4], 16).ok()?;
        let b = u8::from_str_radix(&trimmed[4..6], 16).ok()?;
        Some(Self { r, g, b })
    }

    pub fn to_hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }

    /// Encode as a Python tuple `(r, g, b)` for monty.
    pub fn to_monty(self) -> MontyObject {
        MontyObject::Tuple(vec![
            MontyObject::Int(self.r as i64),
            MontyObject::Int(self.g as i64),
            MontyObject::Int(self.b as i64),
        ])
    }

    /// Decode from a Python tuple/list of three ints. Clamps to 0..=255.
    pub fn from_monty(obj: &MontyObject) -> Option<Self> {
        let triple = match obj {
            MontyObject::Tuple(v) | MontyObject::List(v) => v,
            _ => return None,
        };
        if triple.len() != 3 {
            return None;
        }
        let to_u8 = |o: &MontyObject| -> Option<u8> {
            match o {
                MontyObject::Int(i) => Some((*i).clamp(0, 255) as u8),
                MontyObject::Float(f) => Some(f.round().clamp(0.0, 255.0) as u8),
                MontyObject::Bool(b) => Some(if *b { 255 } else { 0 }),
                _ => None,
            }
        };
        Some(Self {
            r: to_u8(&triple[0])?,
            g: to_u8(&triple[1])?,
            b: to_u8(&triple[2])?,
        })
    }
}

/// Wire-level contract handed to/from the LLM. Two strings, both required.
///
/// `code` MUST contain exactly one top-level definition:
/// `def step(state: tuple[int,int,int], neighbors: list[tuple[int,int,int]],
///           step_index: int) -> tuple[int, int, int]`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CellProgram {
    /// One-sentence English description of the rule this program implements.
    /// Helps humans (and other agents) reason about the system.
    pub rule: String,
    /// Self-contained Python source defining `def step(...)`.
    /// Will be executed by Monty (sandboxed) on every tick.
    pub code: String,
}

impl CellProgram {
    /// JSON schema string used in prompts so the LLM sees the exact contract.
    pub fn self_description() -> String {
        serde_json::to_string_pretty(&schema_for!(Self)).unwrap()
    }

    /// A safe identity program — the cell keeps its color forever.
    /// Useful as the default seed before the LLM proposes anything.
    pub fn identity() -> Self {
        Self {
            rule: "identity: keep current color".to_string(),
            code: r#"def step(state, neighbors, step_index):
    return (state[0], state[1], state[2])
"#
            .to_string(),
        }
    }

    /// Parse-only validation. Microseconds; safe to call on every LLM response.
    pub fn validate(&self) -> Result<(), String> {
        Self::compile_runner(&self.code).map(|_| ())
    }

    /// Run one tick. Returns the cell's next color.
    ///
    /// Errors are surfaced as strings — callers typically log and fall back to
    /// the previous color so the simulation keeps moving.
    #[instrument(skip(self, neighbors), fields(rule = %self.rule, neighbors = neighbors.len(), step_index))]
    pub fn run(&self, state: Rgb, neighbors: &[Rgb], step_index: u32) -> Result<Rgb, String> {
        let runner = Self::compile_runner(&self.code)?;
        let inputs = vec![
            state.to_monty(),
            MontyObject::List(neighbors.iter().map(|n| n.to_monty()).collect()),
            MontyObject::Int(step_index as i64),
        ];
        let result = runner
            .run(inputs, NoLimitTracker, PrintWriter::Disabled)
            .map_err(|e| format!("monty runtime error: {e}"))?;
        Rgb::from_monty(&result)
            .ok_or_else(|| format!("step() must return (r,g,b) ints, got: {result:?}"))
    }

    fn compile_runner(code: &str) -> Result<MontyRun, String> {
        // We append a single trailing call so the script's last expression is
        // `step(...)` — Monty returns the value of the last expression.
        let script = format!("{code}\nstep(state, neighbors, step_index)\n");
        MontyRun::new(
            script,
            "cell_step.py",
            vec![
                "state".to_string(),
                "neighbors".to_string(),
                "step_index".to_string(),
            ],
        )
        .map_err(|e| format!("monty parse error: {e}"))
    }
}

/// System prompt with hard constraints. Keep it short, explicit and example-led
/// — that's what reliably yields well-typed `def step` from current models.
pub fn program_authoring_system_prompt() -> String {
    let schema = CellProgram::self_description();
    format!(
        "You are an LLM Cognitive Unit authoring ONE Python function for a 2D cellular automaton.\n\
         Each cell calls `step` every tick to compute its next RGB color.\n\
         \n\
         OUTPUT — return ONLY a JSON object matching this schema (no prose, no markdown):\n\
         ```json\n{schema}\n```\n\
         \n\
         The `code` field MUST define exactly:\n\
         \n\
         def step(state, neighbors, step_index):\n\
             # state: tuple[int, int, int]                  -- this cell's RGB, channels 0..=255\n\
             # neighbors: list[tuple[int, int, int]]        -- 8 neighbors' RGBs (Moore neighborhood)\n\
             # step_index: int                              -- monotonically increasing tick counter\n\
             # returns: tuple[int, int, int]                -- next RGB, channels 0..=255\n\
             ...\n\
         \n\
         HARD CONSTRAINTS (failure to follow these breaks the simulation):\n\
         - Define a single top-level function called `step` with the exact signature above.\n\
         - Always return a 3-tuple of ints in 0..=255. Clamp aggressively.\n\
         - Pure function: no I/O, no `print`, no `input`, no network, no filesystem, no env.\n\
         - No `import` statements. No `class`. No `match` statements.\n\
         - Allowed builtins only: len, range, sum, min, max, abs, int, float, round, sorted, enumerate, zip, any, all.\n\
         - Keep the body under 30 lines. Pure arithmetic and list comprehensions are fine.\n\
         - Do NOT raise exceptions; handle empty `neighbors` gracefully.\n\
         \n\
         GOAL — produce visually interesting *dynamics*. Pure averaging produces a\n\
         boring heat-equation blur and is DISCOURAGED. Aim for one of:\n\
         - oscillation: use `step_index % k` to flip channels periodically\n\
         - threshold logic: count how many neighbors exceed a threshold (Conway-like)\n\
         - reaction-diffusion: average + nonlinear feedback (e.g. amplify variance)\n\
         - phase shifts: `(c + step_index * 7) % 256` for travelling color waves\n\
         - spatial structure: use the cell's own state as a seed for asymmetric updates\n\
         Combine at least one of these with neighbor information. NO pure linear blends.\n\
         \n\
         EXAMPLE 1 — travelling wave + neighbor coupling:\n\
         {{\n\
           \"rule\": \"travelling phase wave coupled to neighbor minimum\",\n\
           \"code\": \"def step(state, neighbors, step_index):\\n    if not neighbors:\\n        return state\\n    nmin = min(c[2] for c in neighbors)\\n    r = (state[0] + 7 * step_index) % 256\\n    g = (state[1] + (nmin // 8)) % 256\\n    b = (nmin + 3) % 256\\n    return (r, g, b)\\n\"\n\
         }}\n\
         \n\
         EXAMPLE 2 — threshold (Conway-flavoured) on the red channel:\n\
         {{\n\
           \"rule\": \"red channel obeys a 2/3 majority threshold; greens drift\",\n\
           \"code\": \"def step(state, neighbors, step_index):\\n    hot = sum(1 for c in neighbors if c[0] > 127)\\n    r = 255 if hot >= 5 else (state[0] // 2)\\n    g = (state[1] + 13) % 256\\n    b = max(0, state[2] - 1)\\n    return (r, g, b)\\n\"\n\
         }}\n"
    )
}

/// Convenience: ask any of the configured `LLMResolver`s for a program.
/// Picks the first resolver; callers can shard manually if they need more.
pub async fn propose_program_via_resolver(
    resolver: &LLMResolver,
    instruction: &str,
) -> Result<CellProgram, String> {
    let ctx = resolver.to_cognitive_context();
    propose_program(&ctx, instruction).await
}

/// Ask the LLM (via rig) for a `CellProgram`. Validates the returned code with
/// monty's parser; on parse failure, retries once including the parser error
/// as feedback.
#[instrument(skip(ctx), fields(model = %ctx.model_name, base_api = %ctx.base_api, provider = ?ctx.provider))]
pub async fn propose_program(
    ctx: &CognitiveContext,
    instruction: &str,
) -> Result<CellProgram, String> {
    let system_prompt = program_authoring_system_prompt();

    // First attempt
    let first = call_llm_for_program(ctx, &system_prompt, instruction).await?;
    if let Err(parse_err) = first.validate() {
        warn!(error = %parse_err, "cell_program_invalid_retrying");
        // Single retry with the parse error as feedback
        let retry_msg = format!(
            "{instruction}\n\nYour previous attempt failed to parse with: {parse_err}\n\
             Return a corrected JSON object that satisfies the schema."
        );
        let second = call_llm_for_program(ctx, &system_prompt, &retry_msg).await?;
        second
            .validate()
            .map_err(|e| format!("invalid program after retry: {e}"))?;
        debug!(rule = %second.rule, "cell_program_validated_after_retry");
        return Ok(second);
    }

    debug!(rule = %first.rule, "cell_program_validated");
    Ok(first)
}

/// Sampling temperature for program authoring. Read from `LLMCA_TEMP` (default
/// 0.7). Higher values yield more diverse code; lower values are deterministic.
fn sampling_temperature() -> f64 {
    std::env::var("LLMCA_TEMP")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|v| v.clamp(0.0, 2.0))
        .unwrap_or(0.7)
}

async fn call_llm_for_program(
    ctx: &CognitiveContext,
    system_prompt: &str,
    user_message: &str,
) -> Result<CellProgram, String> {
    let temperature = sampling_temperature();
    match ctx.provider {
        LLMProvider::Ollama => {
            let base_url = normalize_ollama_base_url(&ctx.base_api);
            let api_key = ctx.secret_key.trim();
            let client =
                if api_key.is_empty() || api_key == "_" || api_key.eq_ignore_ascii_case("ollama") {
                    ollama::Client::builder()
                        .api_key(Nothing)
                        .base_url(base_url)
                        .build()
                        .map_err(|e| format!("ollama client build failed: {e}"))?
                } else {
                    ollama::Client::builder()
                        .api_key(api_key)
                        .base_url(base_url)
                        .build()
                        .map_err(|e| format!("ollama client build failed: {e}"))?
                };
            let agent = client
                .agent(&ctx.model_name)
                .preamble(system_prompt)
                .temperature(temperature)
                .build();
            let response = agent
                .prompt_typed::<CellProgram>(user_message.to_string())
                .max_turns(1)
                .extended_details()
                .await
                .map_err(|e| format!("ollama typed prompt failed: {e}"))?;
            Ok(response.output)
        }
        LLMProvider::OpenRouter => {
            // We bypass `rig`'s OpenRouter extractor because rig 0.36.0's
            // response parser fails on extra fields some upstream providers
            // (e.g. DeepInfra) include — `cost`, `native_finish_reason`,
            // `cost_details`, etc. A direct OpenAI-compatible call with
            // `response_format=json_object` is both simpler and more robust.
            openrouter_json_object_request(ctx, system_prompt, user_message, temperature).await
        }
    }
}

/// Direct OpenRouter chat-completions call requesting `json_object` output.
/// Returns the parsed `CellProgram`. Surfacing the raw HTTP body in errors so
/// schema drifts are diagnosable from logs alone.
async fn openrouter_json_object_request(
    ctx: &CognitiveContext,
    system_prompt: &str,
    user_message: &str,
    temperature: f64,
) -> Result<CellProgram, String> {
    let url = format!("{}/chat/completions", ctx.base_api.trim_end_matches('/'));

    let body = json!({
        "model": ctx.model_name,
        "temperature": temperature,
        "max_tokens": 1024,
        "response_format": { "type": "json_object" },
        "messages": [
            { "role": "system", "content": system_prompt },
            // We restate the JSON-only constraint on the user turn so the model
            // doesn't emit prose even when its training nudges it that way.
            {
                "role": "user",
                "content": format!(
                    "{user_message}\n\nReturn ONLY a single JSON object with keys `rule` (string) and `code` (string). No markdown, no commentary."
                )
            }
        ]
    });

    let http = HttpClient::new();
    let resp = http
        .post(&url)
        .bearer_auth(ctx.secret_key.trim())
        .header("Content-Type", "application/json")
        // OpenRouter recommends these for analytics; harmless if missing.
        .header("HTTP-Referer", "https://github.com/pinsky-three/llmca")
        .header("X-Title", "LLMCA programmable cells")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("openrouter request failed: {e}"))?;

    let status = resp.status();
    let raw = resp
        .text()
        .await
        .map_err(|e| format!("openrouter body read failed: {e}"))?;

    if !status.is_success() {
        return Err(format!(
            "openrouter http {status}: {body}",
            body = truncate(&raw, 500)
        ));
    }

    let envelope: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("openrouter json parse failed: {e}: {}", truncate(&raw, 500)))?;

    let content = envelope
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            format!(
                "openrouter response missing /choices/0/message/content: {}",
                truncate(&raw, 500)
            )
        })?;

    let cleaned = strip_code_fences(content);
    serde_json::from_str::<CellProgram>(cleaned).map_err(|e| {
        format!(
            "openrouter content not a valid CellProgram JSON ({e}): {}",
            truncate(cleaned, 500)
        )
    })
}

/// Some models still wrap JSON in ```json ... ``` despite instructions.
fn strip_code_fences(s: &str) -> &str {
    let s = s.trim();
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s)
        .trim_start();
    s.strip_suffix("```").unwrap_or(s).trim_end()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut out = s[..max].to_string();
        out.push_str("…");
        out
    }
}

fn normalize_ollama_base_url(api_url: &str) -> String {
    let trimmed = api_url.trim_end_matches('/');
    if let Some(base) = trimmed.strip_suffix("/api/v1") {
        base.to_string()
    } else if let Some(base) = trimmed.strip_suffix("/v1") {
        base.to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_program_runs() {
        let p = CellProgram::identity();
        let out = p
            .run(Rgb::new(10, 20, 30), &[Rgb::new(0, 0, 0)], 0)
            .expect("identity must run");
        assert_eq!(out, Rgb::new(10, 20, 30));
    }

    #[test]
    fn averaging_program_runs() {
        let p = CellProgram {
            rule: "average".into(),
            code: r#"def step(state, neighbors, step_index):
    if not neighbors:
        return state
    n = len(neighbors)
    return (
        sum(c[0] for c in neighbors) // n,
        sum(c[1] for c in neighbors) // n,
        sum(c[2] for c in neighbors) // n,
    )
"#
            .into(),
        };
        p.validate().expect("must parse");
        let out = p
            .run(
                Rgb::new(0, 0, 0),
                &[Rgb::new(100, 100, 100), Rgb::new(200, 200, 200)],
                7,
            )
            .expect("must run");
        assert_eq!(out, Rgb::new(150, 150, 150));
    }

    #[test]
    fn rejects_invalid_python() {
        let p = CellProgram {
            rule: "broken".into(),
            code: "def step(:::)".into(),
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn rgb_round_trip_via_monty() {
        let c = Rgb::new(13, 240, 1);
        let m = c.to_monty();
        assert_eq!(Rgb::from_monty(&m), Some(c));
    }

    #[test]
    fn strip_code_fences_handles_json_block() {
        assert_eq!(strip_code_fences("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_code_fences("```\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_code_fences("{\"a\":1}"), "{\"a\":1}");
    }
}
