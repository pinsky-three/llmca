use chrono::Utc;
// use futures::{stream, StreamExt};
use itertools::Itertools;
use kdam::{tqdm, BarExt};
use petgraph::{stable_graph::StableGraph, Undirected};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use reqwest::Client;

use serde::{Deserialize, Serialize};
// use serde_derive::Serialize;

use crate::{
    system::unit::CognitiveContext,
    system::unit_next::{CognitiveUnitComplex, CognitiveUnitPair, CognitiveUnitWithMemory},
};
use std::{env, fmt::Debug, path::Path, vec};

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
}

impl CognitiveSpaceWithMemory {
    pub fn load_from_json(json: &str) -> Self {
        serde_json::from_str(json).unwrap()
    }

    pub async fn distributed_step(&mut self) {
        let mut nodes = self.graph.clone().node_indices().collect::<Vec<_>>();

        // let mut rng = ThreadRng::default();
        let mut rng = StdRng::from_entropy();

        nodes.shuffle(&mut rng);

        let base_api_urls =
            env::var("OPENAI_API_URL").unwrap_or("http://localhost:11434/v1".to_string());
        let models = env::var("OPENAI_MODEL_NAME").unwrap_or("phi3".to_string());
        let secret_keys = env::var("OPENAI_API_KEY").unwrap_or("ollama".to_string());

        let base_api_urls = base_api_urls
            .split(',')
            .map(|s| s.trim())
            .collect::<Vec<_>>();

        let models = models.split(',').map(|s| s.trim()).collect::<Vec<_>>();

        let secret_keys = secret_keys.split(',').map(|s| s.trim()).collect::<Vec<_>>();

        if base_api_urls.len() != models.len() || models.len() != secret_keys.len() {
            panic!("The number of base_api_urls, models, and secret_keys must be the same.");
        }

        let computation_units = (0..base_api_urls.len())
            .map(|i| {
                let base_api = base_api_urls[i];
                let model_name = models[i];
                let secret_key = secret_keys[i];

                (base_api, model_name, secret_key)
            })
            .collect::<Vec<_>>();

        println!("computation_units: {:?}", computation_units);

        let mut pb: kdam::Bar = tqdm!(total = nodes.len());

        for chunk in nodes.chunks(computation_units.len()) {
            let mut tasks = vec![];

            for (i, &node) in chunk.iter().enumerate() {
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
                    base_api: computation_units[i].0.to_string(),
                    model_name: computation_units[i].1.to_string(),
                    secret_key: computation_units[i].2.to_string(),
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

                let next_state = next_state.unwrap_or_else(|err| {
                    println!("err: {:?}", err);

                    CognitiveUnitComplex::default()
                });

                // unit.state = next_state.calculated_state;
                // unit.feedback = next_state.feedback;

                unit.add_memory(next_state);

                pb.update(1).ok();
            }
        }
    }

    pub async fn distributed_step_with_tasks(
        &mut self,
        resolvers: &[LLMResolver],
        handle: &tokio::runtime::Handle,
    ) {
        let mut nodes = self.graph.clone().node_indices().collect::<Vec<_>>();

        // if self.computing_tasks.is_none() {
        // self.computing_tasks = Some(vec![]); // reset at start new distributed step
        // }

        // let mut rng = ThreadRng::default();
        let mut rng = StdRng::from_entropy();

        nodes.shuffle(&mut rng);

        let computation_units: Vec<(String, String, String)> = resolvers
            .iter()
            .map(|r| (r.api_url.clone(), r.model_name.clone(), r.api_key.clone()))
            .collect();

        // let mut pb: kdam::Bar = tqdm!(total = nodes.len());

        for chunk in nodes.chunks(computation_units.len()) {
            let mut tasks = vec![];

            for (i, &node) in chunk.iter().enumerate() {
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
                    base_api: computation_units[i].0.to_string(),
                    model_name: computation_units[i].1.to_string(),
                    secret_key: computation_units[i].2.to_string(),
                };

                // self.computing_tasks.as_mut().unwrap().push(CognitiveTask {
                //     unit: unit.clone(),
                //     total_units: nodes.len(),
                // });

                tasks.push(handle.spawn(async move {
                    // unit.calculate_next_state(&ctx, neighbors).await

                    unit.calculate_next_complex(&ctx, neighbors).await
                }));
            }

            let next_states = futures::future::join_all(tasks).await;

            for (i, next_state) in next_states.into_iter().enumerate() {
                let node = chunk[i];
                let unit = self.graph.node_weight_mut(node).unwrap();

                let next_state = next_state.unwrap_or_else(|err| {
                    println!("err: {:?}", err);
                    CognitiveUnitComplex::default()
                });

                // unit.state = next_state.calculated_state;
                // unit.feedback = next_state.feedback;

                unit.add_memory(next_state);

                // self.computing_tasks.as_mut().unwrap().pop();

                // pb.update(1).ok();
            }
        }

        // self.computing_tasks = None;
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

pub fn load_llm_resolvers_from_env() -> Vec<LLMResolver> {
    let base_api_urls =
        env::var("OPENAI_API_URL").unwrap_or("http://localhost:11434/v1".to_string());
    let models = env::var("OPENAI_MODEL_NAME").unwrap_or("phi3".to_string());
    let secret_keys = env::var("OPENAI_API_KEY").unwrap_or("ollama".to_string());

    let base_api_urls = base_api_urls
        .split(',')
        .map(|s| s.trim())
        .collect::<Vec<_>>();

    let models = models.split(',').map(|s| s.trim()).collect::<Vec<_>>();

    let secret_keys = secret_keys.split(',').map(|s| s.trim()).collect::<Vec<_>>();

    if base_api_urls.len() != models.len() || models.len() != secret_keys.len() {
        panic!("The number of base_api_urls, models, and secret_keys must be the same.");
    }

    base_api_urls
        .iter()
        .zip(models.iter())
        .zip(secret_keys.iter())
        .map(|((base_api, model_name), secret_key)| LLMResolver {
            api_url: base_api.to_string(),
            model_name: model_name.to_string(),
            api_key: secret_key.to_string(),
        })
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
