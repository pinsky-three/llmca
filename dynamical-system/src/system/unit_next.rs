use chrono::{DateTime, Utc};

use rig::{
    client::{CompletionClient, Nothing},
    completion::TypedPrompt,
    extractor::ExtractorBuilder,
    providers::{ollama, openrouter},
};
use schemars::{schema_for, JsonSchema};

use serde_derive::{Deserialize, Serialize};
use serde_json::json;
use std::time::Instant;
use tracing::{debug, instrument, warn};

use crate::system::unit::{CognitiveContext, LLMProvider};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CognitiveUnitComplex {
    pub timestamp: DateTime<Utc>,
    pub rule: String,
    pub state: String,          // in json format
    pub neighbors: Vec<String>, // in json format
    pub feedback: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CognitiveUnitPair {
    pub rule: String,
    pub state: String,
}

impl CognitiveUnitPair {
    pub fn self_description() -> String {
        serde_json::to_string_pretty(&schema_for!(CognitiveUnitPair)).unwrap()
    }
}

impl Default for CognitiveUnitComplex {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            rule: "".to_string(),
            state: "".to_string(),
            neighbors: vec![],
            feedback: "".to_string(),
        }
    }
}

impl CognitiveUnitComplex {
    pub fn self_description() -> String {
        serde_json::to_string_pretty(&schema_for!(CognitiveUnitComplex)).unwrap()
    }

    pub fn to_pair(&self) -> CognitiveUnitPair {
        CognitiveUnitPair {
            rule: self.rule.clone(),
            state: self.state.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, JsonSchema, Deserialize)]
pub struct CognitiveUnitWithMemory {
    pub memory: Vec<CognitiveUnitComplex>,
    pub memory_size: usize,

    pub position: (usize, usize),
}

impl CognitiveUnitWithMemory {
    pub fn self_description() -> String {
        serde_json::to_string_pretty(&schema_for!(CognitiveUnitWithMemory)).unwrap()
    }

    pub fn new(
        position: (usize, usize),
        memory: Vec<CognitiveUnitComplex>,
        memory_size: usize,
    ) -> Self {
        // let cognitive_unit_description = CognitiveUnitWithMemory::self_description();

        Self {
            position,

            memory,
            memory_size,
        }
    }

    pub fn add_memory(&mut self, memory: CognitiveUnitComplex) {
        self.memory.push(memory);

        if self.memory.len() > self.memory_size {
            self.memory.remove(0);
        }
    }

    #[instrument(skip_all, fields(position = ?self.position, model = %ctx.model_name, base_api = %ctx.base_api, memory = self.memory.len(), neighbors = neighbors.len()))]
    pub async fn calculate_next_complex(
        &self,
        ctx: &CognitiveContext,
        neighbors: Vec<CognitiveUnitPair>,
    ) -> CognitiveUnitComplex {
        let started_at = Instant::now();
        let input_payload = json!({
            "self_memory": self.memory.iter().map(CognitiveUnitComplex::to_pair).collect::<Vec<_>>(),
            "neighbors": neighbors,
        })
        .to_string();

        let pair_description = CognitiveUnitPair::self_description();

        let system_message = [
            "You are an LLM Cognitive Unit. Your task is to choose the next rule and state for this cell from its memory and neighbor states",
            format!("The required output type is `CognitiveUnitPair`: {}", pair_description).as_str(),
            "The `state` should be a compact renderable value, preferably a hexadecimal color like `#ff0000` when the simulation is visualized",
            "Preserve the existing rule unless the memory and neighbors strongly justify a better one",
            "Do not include prose, markdown, comments, or extra fields outside the requested structure",
        ]
        .join(".\n");

        let structured =
            match Self::rig_structured_completion(ctx, &system_message, &input_payload).await {
                Ok(structured) => structured,
                Err(err) => {
                    let feedback = classify_rig_error(&err.to_string());

                    warn!(
                        model = %ctx.model_name,
                        base_api = %ctx.base_api,
                        provider = ?ctx.provider,
                        position = ?self.position,
                        error = %err,
                    feedback = %feedback,
                        "llm_request_failed"
                    );

                    return self.fallback_complex(&neighbors, feedback);
                }
            };

        if tracing::enabled!(target: "llmca::model_response", tracing::Level::DEBUG) {
            debug!(
                target: "llmca::model_response",
                model = %ctx.model_name,
                base_api = %ctx.base_api,
                provider = ?ctx.provider,
                position = ?self.position,
                prompt_tokens = structured.input_tokens,
                completion_tokens = structured.output_tokens,
                total_tokens = structured.total_tokens,
                structured_output = %serde_json::to_string(&structured.pair).unwrap_or_default(),
                "llm_model_response"
            );
        }

        debug!(
            model = %ctx.model_name,
            base_api = %ctx.base_api,
            provider = ?ctx.provider,
            prompt_tokens = structured.input_tokens,
            completion_tokens = structured.output_tokens,
            total_tokens = structured.total_tokens,
            elapsed_ms = started_at.elapsed().as_millis() as u64,
            "llm_request_completed"
        );

        CognitiveUnitComplex {
            timestamp: Utc::now(),
            rule: structured.pair.rule,
            state: structured.pair.state,
            neighbors: neighbors.iter().map(|n| n.state.clone()).collect(),
            feedback: "".to_string(),
        }
    }

    fn fallback_complex(
        &self,
        neighbors: &[CognitiveUnitPair],
        feedback: String,
    ) -> CognitiveUnitComplex {
        let previous = self.memory.last().cloned().unwrap_or_default();

        CognitiveUnitComplex {
            timestamp: Utc::now(),
            rule: previous.rule,
            state: previous.state,
            neighbors: neighbors.iter().map(|n| n.state.clone()).collect(),
            feedback,
        }
    }

    #[instrument(skip_all, fields(model = %ctx.model_name, base_api = %ctx.base_api, provider = ?ctx.provider))]
    async fn rig_structured_completion(
        ctx: &CognitiveContext,
        system_message: &str,
        user_message: &str,
    ) -> Result<StructuredCompletion, Box<dyn std::error::Error + Send + Sync>> {
        match ctx.provider {
            LLMProvider::Ollama => {
                Self::ollama_structured_completion(ctx, system_message, user_message).await
            }
            LLMProvider::OpenRouter => {
                Self::openrouter_structured_completion(ctx, system_message, user_message).await
            }
        }
    }

    async fn ollama_structured_completion(
        ctx: &CognitiveContext,
        system_message: &str,
        user_message: &str,
    ) -> Result<StructuredCompletion, Box<dyn std::error::Error + Send + Sync>> {
        let client = build_ollama_client(ctx)?;
        let agent = client
            .agent(&ctx.model_name)
            .preamble(system_message)
            .temperature(0.0)
            .build();

        let response = agent
            .prompt_typed::<CognitiveUnitPair>(user_message.to_string())
            .max_turns(1)
            .extended_details()
            .await?;

        Ok(StructuredCompletion::new(response.output, response.usage))
    }

    async fn openrouter_structured_completion(
        ctx: &CognitiveContext,
        system_message: &str,
        user_message: &str,
    ) -> Result<StructuredCompletion, Box<dyn std::error::Error + Send + Sync>> {
        let client = openrouter::Client::builder()
            .api_key(ctx.secret_key.as_str())
            .base_url(ctx.base_api.as_str())
            .build()?;

        let model = client.completion_model(&ctx.model_name).with_strict_tools();
        let extractor = ExtractorBuilder::<_, CognitiveUnitPair>::new(model)
            .preamble(system_message)
            .max_tokens(512)
            .retries(1)
            .build();

        let response = extractor
            .extract_with_usage(user_message.to_string())
            .await?;

        Ok(StructuredCompletion::new(response.data, response.usage))
    }
}

struct StructuredCompletion {
    pair: CognitiveUnitPair,
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
}

impl StructuredCompletion {
    fn new(pair: CognitiveUnitPair, usage: rig::completion::Usage) -> Self {
        Self {
            pair,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

fn build_ollama_client(
    ctx: &CognitiveContext,
) -> Result<ollama::Client, Box<dyn std::error::Error + Send + Sync>> {
    let base_url = normalize_ollama_base_url(&ctx.base_api);
    let api_key = ctx.secret_key.trim();

    let client = if api_key.is_empty() || api_key == "_" || api_key.eq_ignore_ascii_case("ollama") {
        ollama::Client::builder()
            .api_key(Nothing)
            .base_url(base_url)
            .build()?
    } else {
        ollama::Client::builder()
            .api_key(api_key)
            .base_url(base_url)
            .build()?
    };

    Ok(client)
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

fn classify_rig_error(error: &str) -> String {
    if error.contains("DeserializationError")
        || error.contains("No data extracted")
        || error.contains("EmptyResponse")
    {
        format!("Structured output failed: {error}")
    } else {
        format!("LLM request failed: {error}")
    }
}
