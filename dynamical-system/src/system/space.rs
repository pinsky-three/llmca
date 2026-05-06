use chrono::Utc;
// use futures::{stream, StreamExt};
use itertools::Itertools;
use petgraph::{stable_graph::StableGraph, Undirected};
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use reqwest::Client;

use serde_derive::{Deserialize, Serialize};

use crate::{
    system::telemetry::StepTelemetry,
    system::unit::{CognitiveContext, LLMProvider},
    system::unit_next::{CognitiveUnitComplex, CognitiveUnitPair, CognitiveUnitWithMemory},
};
use std::{
    collections::HashSet,
    env,
    fmt::Debug,
    path::Path,
    time::{Duration, Instant},
    vec,
};
use tracing::{debug, info, instrument, warn};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CognitiveTask {
    pub unit: CognitiveUnitWithMemory,
    pub total_units: usize,
    // features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CognitiveSpaceWithMemory {
    // _rule: Box<R>,
    graph: StableGraph<CognitiveUnitWithMemory, (), Undirected>,
    // computing_tasks: Option<Vec<CognitiveTask>>,
}

pub trait CognitiveRule {
    fn compile_prompt(&self) -> String;
}

impl CognitiveRule for () {
    fn compile_prompt(&self) -> String {
        "".to_string()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageModelRule {
    prompt: String,
    features: Vec<String>,
}

impl MessageModelRule {
    pub fn new(prompt: String, features: Vec<String>) -> Self {
        Self { prompt, features }
    }

    pub fn with_feature(&mut self, feature: String) -> Vec<String> {
        self.features.push(feature);
        self.features.clone()
    }
}

impl Default for MessageModelRule {
    fn default() -> Self {
        Self {
            prompt: "You're a LLM Cognitive Unit and your unique task is to respond with your next state based on the state of your neighbors in json format based on:".to_string(),

            features: vec!["rule".to_string(), "state".to_string(), "neighbors".to_string()],
        }
    }
}

impl CognitiveRule for MessageModelRule {
    fn compile_prompt(&self) -> String {
        self.prompt.clone()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResolver {
    api_url: String,
    api_key: String,
    model_name: String,
    #[serde(default)]
    provider: Option<LLMProvider>,
}

impl LLMResolver {
    fn provider(&self) -> LLMProvider {
        self.provider
            .clone()
            .unwrap_or_else(|| LLMProvider::infer_from_api_url(&self.api_url))
    }
}

impl CognitiveSpaceWithMemory {
    pub fn load_from_json(json: &str) -> Self {
        serde_json::from_str(json).unwrap()
    }

    #[instrument(skip_all, fields(units = self.graph.node_count(), resolvers = resolvers.len()))]
    pub async fn distributed_step(&mut self, resolvers: &[LLMResolver]) -> StepTelemetry {
        let started_at = Instant::now();
        let mut nodes = self.graph.clone().node_indices().collect::<Vec<_>>();
        let mut telemetry = StepTelemetry::new(nodes.len(), resolvers.len());

        // Seed a Send-able StdRng from the thread-local RNG inside a tight
        // scope so the !Send `ThreadRng` is dropped before any `.await` and
        // the resulting future stays `Send` (required by `tokio::spawn` and
        // poem's `#[OpenApi]` handlers).
        let mut rng = {
            let mut thread_rng = ThreadRng::default();
            StdRng::from_rng(&mut thread_rng)
        };

        nodes.shuffle(&mut rng);

        let base_api_urls = resolvers
            .iter()
            .map(|r| r.api_url.clone())
            .collect::<Vec<_>>();

        let models = resolvers
            .iter()
            .map(|r| r.model_name.clone())
            .collect::<Vec<_>>();

        let secret_keys = resolvers
            .iter()
            .map(|r| r.api_key.clone())
            .collect::<Vec<_>>();

        // let base_api_urls = base_api_urls
        //     .split(',')
        //     .map(|s| s.trim())
        //     .collect::<Vec<_>>();

        // let models = models.split(',').map(|s| s.trim()).collect::<Vec<_>>();

        // let secret_keys = secret_keys.split(',').map(|s| s.trim()).collect::<Vec<_>>();

        if base_api_urls.len() != models.len() || models.len() != secret_keys.len() {
            panic!("The number of base_api_urls, models, and secret_keys must be the same.");
        }

        let providers = resolvers
            .iter()
            .map(|resolver| resolver.provider())
            .collect::<Vec<_>>();

        let computation_units = (0..base_api_urls.len())
            .map(|i| {
                let base_api = base_api_urls[i].clone();
                let model_name = models[i].clone();
                let secret_key = secret_keys[i].clone();
                let provider = providers[i].clone();

                (base_api, model_name, secret_key, provider)
            })
            .collect::<Vec<_>>();

        if computation_units.is_empty() {
            warn!("distributed_step_skipped_no_resolvers");
            telemetry.finish(started_at.elapsed(), self.unique_state_count());
            return telemetry;
        }

        let chunk_width = computation_units.len();
        info!(
            units_total = nodes.len(),
            resolver_count = computation_units.len(),
            max_in_flight_requests = chunk_width,
            "distributed_step_started"
        );

        for chunk in nodes.chunks(chunk_width) {
            let chunk_started_at = Instant::now();
            telemetry.record_chunk();
            let mut tasks = vec![];

            for (i, &node) in chunk.iter().enumerate() {
                let resolver_index = i % computation_units.len();
                let neighbors = self
                    .graph
                    .neighbors(node)
                    .map(|neighbor| {
                        let neighbor_unit = self.graph.node_weight(neighbor).unwrap();

                        // (
                        // format!("n_{}", neighbor.index()),
                        neighbor_unit.memory.last().unwrap().to_pair()
                        // )
                    })
                    .collect::<Vec<_>>();

                let unit = self.graph.node_weight_mut(node).unwrap().clone();
                let ctx = CognitiveContext {
                    client: Box::new(Client::new()),
                    base_api: computation_units[resolver_index].0.to_string(),
                    model_name: computation_units[resolver_index].1.to_string(),
                    secret_key: computation_units[resolver_index].2.to_string(),
                    provider: computation_units[resolver_index].3.clone(),
                };

                tasks.push(tokio::spawn(async move {
                    // unit.calculate_next_state(&ctx, neighbors).await

                    unit.calculate_next_complex(&ctx, neighbors).await
                }));
            }

            let next_states = futures::future::join_all(tasks).await;

            for (i, next_state) in next_states.into_iter().enumerate() {
                let node = chunk[i];
                let unit = self.graph.node_weight_mut(node).unwrap();

                let next_state = match next_state {
                    Ok(next_state) => next_state,
                    Err(err) => {
                        warn!(error = ?err, "llm_task_join_failed");

                        let previous = unit.memory.last().cloned().unwrap_or_default();

                        CognitiveUnitComplex {
                            timestamp: Utc::now(),
                            rule: previous.rule,
                            state: previous.state,
                            neighbors: vec![],
                            feedback: format!("LLM request failed: task join error: {err}"),
                        }
                    }
                };

                // unit.state = next_state.calculated_state;
                // unit.feedback = next_state.feedback;

                telemetry.record_unit(&next_state);
                unit.add_memory(next_state);
            }

            log_slow_chunk(
                chunk.len(),
                chunk_started_at.elapsed(),
                telemetry.chunks,
                telemetry.units_completed,
                telemetry.units_total,
            );
        }

        telemetry.finish(started_at.elapsed(), self.unique_state_count());
        info!(
            units_total = telemetry.units_total,
            units_completed = telemetry.units_completed,
            resolver_count = telemetry.resolver_count,
            chunks = telemetry.chunks,
            llm_failures = telemetry.llm_failures,
            parse_failures = telemetry.parse_failures,
            unique_states = telemetry.unique_states,
            elapsed_ms = telemetry.elapsed_ms,
            "distributed_step_completed"
        );

        telemetry
    }

    pub async fn distributed_step_with_tasks(
        &mut self,
        resolvers: &[LLMResolver],
        _handle: &tokio::runtime::Handle,
    ) -> StepTelemetry {
        self.distributed_step(resolvers).await
    }

    pub fn generate_graph(&self) -> StableGraph<CognitiveUnitWithMemory, (), Undirected> {
        self.graph.clone()
    }

    pub fn get_units(&self) -> Vec<CognitiveUnitWithMemory> {
        self.graph.node_weights().cloned().collect()
    }

    pub fn set_unit(&mut self, i: usize, j: usize, unit: CognitiveUnitWithMemory) {
        let xy_to_index = |i: usize, j: usize| -> usize { i * j + j };

        let node = self.graph.node_indices().nth(xy_to_index(i, j)).unwrap();

        let internal_unit = self.graph.node_weight_mut(node).unwrap();

        internal_unit.memory = unit.memory;
        internal_unit.memory_size = unit.memory_size;
        internal_unit.position = unit.position;
    }

    pub fn serialize_in_pretty_json(&self) -> String {
        serde_json::to_string_pretty(&self).unwrap()
    }

    fn unique_state_count(&self) -> usize {
        self.graph
            .node_weights()
            .filter_map(|unit| unit.memory.last().map(|memory| memory.state.as_str()))
            .collect::<HashSet<_>>()
            .len()
    }

    // pub fn computing_tasks(&self) -> Vec<CognitiveTask> {
    //     if self.computing_tasks.is_none() {
    //         return vec![];
    //     }

    //     self.computing_tasks.to_owned().unwrap().clone()
    // }
}

pub fn build_lattice_with_memory(
    n: usize,
    m: usize,
    memory_size: usize,
    cognitive_unit_init_state: impl Fn((usize, usize)) -> CognitiveUnitPair,
) -> CognitiveSpaceWithMemory {
    let xy_to_index = |i: usize, j: usize| -> usize { i * m + j };

    // let mut rng = thread_rng();

    let mut graph =
        StableGraph::<CognitiveUnitWithMemory, (), Undirected>::with_capacity(n * m, 8 * n * m);

    let (nodes, positions): (Vec<_>, Vec<_>) = (0..n)
        .cartesian_product(0..m)
        .map(|position| {
            // let state = self.initial_states.choose(&mut rng).unwrap().to_owned();
            // let rule = self.rule.compile_prompt();

            // let unit = CognitiveUnitWithMemory {
            //     rule,
            //     state,
            //     position,
            //     feedback: None,
            // };

            let first_unit = cognitive_unit_init_state(position);

            let unit = CognitiveUnitWithMemory::new(
                position,
                vec![CognitiveUnitComplex {
                    timestamp: Utc::now(),
                    rule: first_unit.rule.clone(),
                    state: first_unit.state.clone(),
                    neighbors: vec![],
                    feedback: "".to_string(),
                }],
                memory_size,
            );

            (graph.add_node(unit.clone()), position)
        })
        .unzip();

    positions.iter().for_each(|&(i, j)| {
        let i_s1 = if i.overflowing_sub(1).1 { n - 1 } else { i };
        let j_s1 = if j.overflowing_sub(1).1 { m - 1 } else { j };

        let i_a1 = (i + 1) % n;
        let j_a1 = (j + 1) % m;

        let n_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_a1, j)]);
        let ne_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_a1, j_a1)]);
        let e_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i, j_a1)]);
        let se_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_s1, j_a1)]);
        let s_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_s1, j)]);
        let sw_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_s1, j_s1)]);
        let w_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i, j_s1)]);
        let nw_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_a1, j_s1)]);

        if !graph.contains_edge(n_edge.0, n_edge.1) && n_edge.0 != n_edge.1 {
            graph.add_edge(n_edge.0, n_edge.1, ());
        }

        if !graph.contains_edge(ne_edge.0, ne_edge.1) && ne_edge.0 != ne_edge.1 {
            graph.add_edge(ne_edge.0, ne_edge.1, ());
        }

        if !graph.contains_edge(e_edge.0, e_edge.1) && e_edge.0 != e_edge.1 {
            graph.add_edge(e_edge.0, e_edge.1, ());
        }

        if !graph.contains_edge(se_edge.0, se_edge.1) && se_edge.0 != se_edge.1 {
            graph.add_edge(se_edge.0, se_edge.1, ());
        }

        if !graph.contains_edge(s_edge.0, s_edge.1) && s_edge.0 != s_edge.1 {
            graph.add_edge(s_edge.0, s_edge.1, ());
        }

        if !graph.contains_edge(sw_edge.0, sw_edge.1) && sw_edge.0 != sw_edge.1 {
            graph.add_edge(sw_edge.0, sw_edge.1, ());
        }

        if !graph.contains_edge(w_edge.0, w_edge.1) && w_edge.0 != w_edge.1 {
            graph.add_edge(w_edge.0, w_edge.1, ());
        }

        if !graph.contains_edge(nw_edge.0, nw_edge.1) && nw_edge.0 != nw_edge.1 {
            graph.add_edge(nw_edge.0, nw_edge.1, ());
        }
    });

    CognitiveSpaceWithMemory {
        // computing_tasks: Option::None,
        graph,
    }
}

fn log_slow_chunk(
    chunk_units: usize,
    elapsed: Duration,
    chunk_index: usize,
    units_completed: usize,
    units_total: usize,
) {
    let elapsed_ms = elapsed.as_millis() as u64;

    if elapsed_ms >= 5_000 {
        info!(
            chunk_index,
            chunk_units,
            units_completed,
            units_total,
            elapsed_ms,
            "distributed_step_chunk_completed"
        );
    } else {
        debug!(
            chunk_index,
            chunk_units,
            units_completed,
            units_total,
            elapsed_ms,
            "distributed_step_chunk_completed"
        );
    }
}

pub fn load_llm_resolvers_from_env() -> Vec<LLMResolver> {
    let base_api_urls =
        env::var("OPENAI_API_URL").unwrap_or("http://localhost:11434/v1".to_string());
    let models = env::var("OPENAI_MODEL_NAME").unwrap_or("phi3".to_string());
    let secret_keys = env::var("OPENAI_API_KEY").unwrap_or("ollama".to_string());
    let providers = env::var("OPENAI_PROVIDER").ok();

    let base_api_urls = base_api_urls
        .split(',')
        .map(|s| s.trim())
        .collect::<Vec<_>>();

    let models = models.split(',').map(|s| s.trim()).collect::<Vec<_>>();

    let secret_keys = secret_keys.split(',').map(|s| s.trim()).collect::<Vec<_>>();
    let providers = providers.map(|providers| {
        providers
            .split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<_>>()
    });

    if base_api_urls.len() != models.len() || models.len() != secret_keys.len() {
        panic!("The number of base_api_urls, models, and secret_keys must be the same.");
    }

    if let Some(providers) = &providers {
        if providers.len() != base_api_urls.len() {
            panic!("The number of providers must match the number of base_api_urls.");
        }
    }

    base_api_urls
        .iter()
        .enumerate()
        .zip(models.iter())
        .zip(secret_keys.iter())
        .map(
            |(((index, base_api), model_name), secret_key)| LLMResolver {
                api_url: base_api.to_string(),
                model_name: model_name.to_string(),
                api_key: secret_key.to_string(),
                provider: providers
                    .as_ref()
                    .and_then(|providers| LLMProvider::parse(&providers[index]))
                    .or_else(|| Some(LLMProvider::infer_from_api_url(base_api))),
            },
        )
        .collect()
}

pub fn load_llm_resolvers_from_toml<P: AsRef<Path>>(path: P) -> Vec<LLMResolver> {
    let toml = std::fs::read_to_string(path).unwrap();

    let resolvers: TomlConfig = toml::from_str(&toml).unwrap();

    resolvers.resolvers
}

#[derive(Serialize, Deserialize, Debug)]
struct TomlConfig {
    resolvers: Vec<LLMResolver>,
}
