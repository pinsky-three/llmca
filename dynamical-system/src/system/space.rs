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
use std::{env, fmt::Debug};

// #[derive(Debug, Clone)]
// pub struct CognitiveSpace<R>
// where
//     R: CognitiveRule,
// {
//     _rule: Box<R>,
//     graph: StableGraph<CognitiveUnit, (), Undirected>,
// }

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
    computing_tasks: Option<Vec<CognitiveTask>>,
}

pub trait CognitiveRule {
    fn compile_prompt(&self) -> String;
    // fn get_rule_prompt(&self, Vec<String>) -> String;
    // fn get_rule_prompt(&self, fn(Vec<String>)->String) -> String;
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

// pub struct VonNeumannLatticeCognitiveSpace<R>
// where
//     R: CognitiveRule + Debug + Clone,
// {
//     rule: Box<R>,
//     initial_states: Vec<Vec<String>>,
// }

// impl<R> VonNeumannLatticeCognitiveSpace<R>
// where
//     R: CognitiveRule + Debug + Clone,
// {
//     pub fn new(rule: R, initial_states: Vec<Vec<String>>) -> Self {
//         Self {
//             rule: Box::new(rule),
//             initial_states,
//         }
//     }

//     pub fn build_lattice(&self, n: usize, m: usize) -> CognitiveSpace<R> {
//         let xy_to_index = |i: usize, j: usize| -> usize { i * m + j };

//         let mut rng = thread_rng();

//         let mut graph =
//             StableGraph::<CognitiveUnit, (), Undirected>::with_capacity(n * m, 8 * n * m);

//         let (nodes, positions): (Vec<_>, Vec<_>) = (0..n)
//             .cartesian_product(0..m)
//             .map(|position| {
//                 let state = self.initial_states.choose(&mut rng).unwrap().to_owned();
//                 let rule = self.rule.compile_prompt();
//                 let unit = CognitiveUnit {
//                     rule,
//                     state,
//                     position,
//                     feedback: None,
//                 };

//                 (graph.add_node(unit.clone()), position)
//             })
//             .unzip();

//         positions.iter().for_each(|&(i, j)| {
//             let i_s1 = i.overflowing_sub(1).1.then(|| n - 1).unwrap_or(i);
//             let j_s1 = j.overflowing_sub(1).1.then(|| m - 1).unwrap_or(j);

//             let i_a1 = (i + 1) % n;
//             let j_a1 = (j + 1) % m;

//             let n_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_a1, j)]);
//             let ne_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_a1, j_a1)]);
//             let e_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i, j_a1)]);
//             let se_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_s1, j_a1)]);
//             let s_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_s1, j)]);
//             let sw_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_s1, j_s1)]);
//             let w_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i, j_s1)]);
//             let nw_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_a1, j_s1)]);

//             if !graph.contains_edge(n_edge.0, n_edge.1) && n_edge.0 != n_edge.1 {
//                 graph.add_edge(n_edge.0, n_edge.1, ());
//             }

//             if !graph.contains_edge(ne_edge.0, ne_edge.1) && ne_edge.0 != ne_edge.1 {
//                 graph.add_edge(ne_edge.0, ne_edge.1, ());
//             }

//             if !graph.contains_edge(e_edge.0, e_edge.1) && e_edge.0 != e_edge.1 {
//                 graph.add_edge(e_edge.0, e_edge.1, ());
//             }

//             if !graph.contains_edge(se_edge.0, se_edge.1) && se_edge.0 != se_edge.1 {
//                 graph.add_edge(se_edge.0, se_edge.1, ());
//             }

//             if !graph.contains_edge(s_edge.0, s_edge.1) && s_edge.0 != s_edge.1 {
//                 graph.add_edge(s_edge.0, s_edge.1, ());
//             }

//             if !graph.contains_edge(sw_edge.0, sw_edge.1) && sw_edge.0 != sw_edge.1 {
//                 graph.add_edge(sw_edge.0, sw_edge.1, ());
//             }

//             if !graph.contains_edge(w_edge.0, w_edge.1) && w_edge.0 != w_edge.1 {
//                 graph.add_edge(w_edge.0, w_edge.1, ());
//             }

//             if !graph.contains_edge(nw_edge.0, nw_edge.1) && nw_edge.0 != nw_edge.1 {
//                 graph.add_edge(nw_edge.0, nw_edge.1, ());
//             }
//         });

//         CognitiveSpace {
//             _rule: self.rule.clone(),
//             graph,
//         }
//     }
// }

pub struct LLMResolver {
    base_api: String,
    model_name: String,
    secret_key: String,
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

    pub async fn distributed_step_with_tasks(&mut self, resolvers: Vec<LLMResolver>) {
        let mut nodes = self.graph.clone().node_indices().collect::<Vec<_>>();

        // let mut rng = ThreadRng::default();
        let mut rng = StdRng::from_entropy();

        nodes.shuffle(&mut rng);

        // let base_api_urls =
        //     env::var("OPENAI_API_URL").unwrap_or("http://localhost:11434/v1".to_string());
        // let models = env::var("OPENAI_MODEL_NAME").unwrap_or("phi3".to_string());
        // let secret_keys = env::var("OPENAI_API_KEY").unwrap_or("ollama".to_string());

        // let base_api_urls = base_api_urls
        //     .split(',')
        //     .map(|s| s.trim())
        //     .collect::<Vec<_>>();

        // let models = models.split(',').map(|s| s.trim()).collect::<Vec<_>>();

        // let secret_keys = secret_keys.split(',').map(|s| s.trim()).collect::<Vec<_>>();

        // if base_api_urls.len() != models.len() || models.len() != secret_keys.len() {
        //     panic!("The number of base_api_urls, models, and secret_keys must be the same.");
        // }

        // let (base_api_urls, models, secret_keys) = resolvers.iter().fold(
        //     (vec![], vec![], vec![]),
        //     |(mut base_api_urls, mut models, mut secret_keys), resolver| {
        //         base_api_urls.push(resolver.base_api.clone());
        //         models.push(resolver.model_name.clone());
        //         secret_keys.push(resolver.secret_key.clone());

        //         (base_api_urls, models, secret_keys)
        //     },
        // );

        // let computation_units = (0..base_api_urls.len())
        //     .map(|i| {
        //         let base_api = base_api_urls[i];
        //         let model_name = models[i];
        //         let secret_key = secret_keys[i];

        //         (base_api, model_name, secret_key)
        //     })
        //     .collect::<Vec<_>>();

        // println!("computation_units: {:?}", computation_units);

        let computation_units: Vec<(String, String, String)> = resolvers
            .iter()
            .map(|r| {
                (
                    r.base_api.clone(),
                    r.model_name.clone(),
                    r.secret_key.clone(),
                )
            })
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

                self.computing_tasks.as_mut().unwrap().push(CognitiveTask {
                    unit: unit.clone(),
                    total_units: nodes.len(),
                });

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

                self.computing_tasks.as_mut().unwrap().pop();

                // pb.update(1).ok();
            }
        }
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

    pub fn computing_tasks(&self) -> Vec<CognitiveTask> {
        self.computing_tasks.to_owned().unwrap().clone()
    }
}

// impl<R> CognitiveSpace<R>
// where
//     R: CognitiveRule + Debug + Clone,
// {
//     pub async fn distributed_step(&mut self) {
//         let nodes = self.graph.clone().node_indices().collect::<Vec<_>>();

//         let base_api_urls =
//             env::var("OPENAI_API_URL").unwrap_or("http://localhost:11434/v1".to_string());
//         let models = env::var("OPENAI_MODEL_NAME").unwrap_or("phi3".to_string());
//         let secret_keys = env::var("OPENAI_API_KEY").unwrap_or("ollama".to_string());

//         let base_api_urls = base_api_urls
//             .split(',')
//             .map(|s| s.trim())
//             .collect::<Vec<_>>();
//         let models = models.split(',').map(|s| s.trim()).collect::<Vec<_>>();
//         let secret_keys = secret_keys.split(',').map(|s| s.trim()).collect::<Vec<_>>();

//         if base_api_urls.len() != models.len() || models.len() != secret_keys.len() {
//             panic!("The number of base_api_urls, models, and secret_keys must be the same.");
//         }

//         let computation_units = (0..base_api_urls.len())
//             .map(|i| {
//                 let base_api = base_api_urls[i];
//                 let model_name = models[i];
//                 let secret_key = secret_keys[i];

//                 (base_api, model_name, secret_key)
//             })
//             .collect::<Vec<_>>();

//         println!("computation_units: {:?}", computation_units);

//         let mut pb: kdam::Bar = tqdm!(total = nodes.len());

//         for chunk in nodes.chunks(computation_units.len()) {
//             let mut tasks = vec![];

//             for (i, &node) in chunk.iter().enumerate() {
//                 let neighbors = self
//                     .graph
//                     .neighbors(node)
//                     .map(|neighbor| {
//                         let neighbor_unit = self.graph.node_weight(neighbor).unwrap();

//                         (
//                             format!("n_{}", neighbor.index()),
//                             neighbor_unit.state.clone(),
//                         )
//                     })
//                     .collect::<Vec<_>>();

//                 let unit = self.graph.node_weight_mut(node).unwrap().clone();
//                 let ctx = CognitiveContext {
//                     client: Box::new(Client::new()),
//                     base_api: computation_units[i].0.to_string(),
//                     model_name: computation_units[i].1.to_string(),
//                     secret_key: computation_units[i].2.to_string(),
//                 };

//                 tasks.push(tokio::spawn(async move {
//                     unit.calculate_next_state(&ctx, neighbors).await
//                 }));
//             }

//             let next_states = futures::future::join_all(tasks).await;

//             for (i, next_state) in next_states.into_iter().enumerate() {
//                 let node = chunk[i];
//                 let unit = self.graph.node_weight_mut(node).unwrap();

//                 let next_state = next_state.unwrap();

//                 unit.state = next_state.calculated_state;
//                 unit.feedback = next_state.feedback;

//                 pb.update(1).ok();
//             }
//         }
//     }

//     pub fn print_nodes_state(&self) {
//         self.graph.node_indices().for_each(|node| {
//             let unit = self.graph.node_weight(node).unwrap();
//             println!("unit: {:?}", unit.state);
//         });
//     }

//     pub fn generate_graph(&self) -> StableGraph<CognitiveUnit, (), Undirected> {
//         self.graph.clone()
//     }

//     pub fn get_units(&self) -> Vec<CognitiveUnit> {
//         self.graph.node_weights().cloned().collect()
//     }

//     pub fn set_unit(&mut self, i: usize, j: usize, unit: CognitiveUnit) {
//         let xy_to_index = |i: usize, j: usize| -> usize { i * j + j };

//         let node = self.graph.node_indices().nth(xy_to_index(i, j)).unwrap();

//         let internal_unit = self.graph.node_weight_mut(node).unwrap();

//         internal_unit.state = unit.state;
//         internal_unit.rule = unit.rule;
//         internal_unit.position = unit.position;
//         internal_unit.feedback = unit.feedback;
//     }
// }

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
        let i_s1 = i.overflowing_sub(1).1.then(|| n - 1).unwrap_or(i);
        let j_s1 = j.overflowing_sub(1).1.then(|| m - 1).unwrap_or(j);

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
        computing_tasks: Option::None,
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
            base_api: base_api.to_string(),
            model_name: model_name.to_string(),
            secret_key: secret_key.to_string(),
        })
        .collect()
}
