use itertools::Itertools;
use kdam::tqdm;
use petgraph::{stable_graph::StableGraph, Undirected};
use rand::{seq::SliceRandom, thread_rng};

use crate::unit::CognitiveUnit;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct CognitiveSpace<R>
where
    R: CognitiveRule,
{
    // units: Vec<CognitiveUnit>,
    // connections: Vec<(usize, usize)>,
    _rule: Box<R>,
    graph: StableGraph<CognitiveUnit, (), Undirected>,
}

impl<R> CognitiveSpace<R>
where
    R: CognitiveRule + Debug + Clone,
{
    // pub fn new(rule: Box<R>) -> Self {
    //     Self {
    //         units: vec![],
    //         connections: vec![],
    //         rule,
    //     }
    // }

    // fn add_unit(&mut self, unit: CognitiveUnit) {
    //     self.units.push(unit);
    // }

    // fn add_connection(&mut self, from: usize, to: usize) {
    //     self.connections.push((from, to));
    // }
}

pub trait CognitiveRule {
    fn get_rule_prompt(&self) -> String;
    // fn get_rule_prompt(&self, Vec<String>) -> String;
    // fn get_rule_prompt(&self, fn(Vec<String>)->String) -> String;
}

#[derive(Debug, Clone)]
pub struct MessageModelRule {
    prompt: String,
}

impl MessageModelRule {
    pub fn new(prompt: String) -> Self {
        Self { prompt }
    }
}

impl CognitiveRule for MessageModelRule {
    fn get_rule_prompt(&self) -> String {
        self.prompt.clone()
    }
}

pub struct VonNeumannLatticeCognitiveSpace<R>
where
    R: CognitiveRule + Debug + Clone,
{
    rule: Box<R>,
    initial_states: Vec<String>,
}

impl<R> VonNeumannLatticeCognitiveSpace<R>
where
    R: CognitiveRule + Debug + Clone,
{
    pub fn new(rule: R, initial_states: Vec<String>) -> Self {
        Self {
            rule: Box::new(rule),
            initial_states,
        }
    }

    pub fn build_lattice(&self, n: usize, m: usize) -> CognitiveSpace<R> {
        // let mut space = CognitiveSpace::new(Box::new(rule.clone()));
        let xy_to_index = |i: usize, j: usize| -> usize { i * m + j };

        let mut rng = thread_rng();

        let mut graph =
            StableGraph::<CognitiveUnit, (), Undirected>::with_capacity(n * m, 4 * n * m);

        let (nodes, positions): (Vec<_>, Vec<_>) = (0..n)
            .cartesian_product(0..m)
            .map(|position| {
                let state = vec![self.initial_states.choose(&mut rng).unwrap().to_string()];
                let rule = self.rule.get_rule_prompt();
                let unit = CognitiveUnit {
                    rule,
                    state,
                    position,
                };

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

        CognitiveSpace {
            _rule: self.rule.clone(),
            graph,
        }
    }
}

impl<R> CognitiveSpace<R>
where
    R: CognitiveRule + Debug + Clone,
{
    pub fn sync_step(&mut self) {
        let nodes = self.graph.clone().node_indices().collect::<Vec<_>>();

        for i in tqdm!(0..nodes.len()) {
            let node = nodes[i];
            let neighbors = self
                .graph
                .neighbors(node)
                .map(|neighbor| {
                    let neighbor_unit = self.graph.node_weight(neighbor).unwrap();

                    (
                        format!("n_{}", neighbor.index()),
                        neighbor_unit.state.clone(),
                    )
                })
                .collect();

            let unit = self.graph.node_weight_mut(node).unwrap();
            let next_state = unit.calculate_next_state(neighbors);

            // println!("next_state: {:?}", next_state);

            unit.state = next_state;
        }
        // .for_each(|node| {

        // });
    }

    pub fn print_nodes_state(&self) {
        self.graph.node_indices().for_each(|node| {
            let unit = self.graph.node_weight(node).unwrap();
            println!("unit: {:?}", unit.state);
        });
    }

    pub fn generate_graph(&self) -> StableGraph<CognitiveUnit, (), Undirected> {
        self.graph.clone()
    }

    pub fn get_units(&self) -> Vec<CognitiveUnit> {
        self.graph.node_weights().cloned().collect()
    }
}
