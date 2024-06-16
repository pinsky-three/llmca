use itertools::Itertools;
use petgraph::{
    graph::NodeIndex,
    matrix_graph::{MatrixGraph, UnMatrix},
    stable_graph::{StableGraph, StableUnGraph},
    visit::IntoEdges,
    Undirected,
};

use crate::unit::CognitiveUnit;
use std::{fmt::Debug, marker::PhantomData};

#[derive(Debug, Clone)]
pub struct CognitiveSpace<R>
where
    R: CognitiveRule,
{
    // units: Vec<CognitiveUnit>,
    // connections: Vec<(usize, usize)>,
    rule: Box<R>,
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
    _rule: PhantomData<R>,
}

impl<R> VonNeumannLatticeCognitiveSpace<R>
where
    R: CognitiveRule + Debug + Clone,
{
    pub fn new_lattice(n: usize, m: usize, rule: R) -> CognitiveSpace<R> {
        // let mut space = CognitiveSpace::new(Box::new(rule.clone()));
        let xy_to_index = |i: usize, j: usize| -> usize { i * m + j };

        let mut graph =
            StableGraph::<CognitiveUnit, (), Undirected>::with_capacity(n * m, 4 * n * m);

        let (nodes, positions): (Vec<_>, Vec<_>) = (0..n)
            .cartesian_product(0..m)
            .map(|position| {
                let state = vec!["0".to_string()];
                let rule = rule.get_rule_prompt();
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
            let e_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i, j_a1)]);
            let s_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i_s1, j)]);
            let w_edge = (nodes[xy_to_index(i, j)], nodes[xy_to_index(i, j_s1)]);

            if !graph.contains_edge(n_edge.0, n_edge.1) && n_edge.0 != n_edge.1 {
                graph.add_edge(n_edge.0, n_edge.1, ());
            }

            if !graph.contains_edge(e_edge.0, e_edge.1) && e_edge.0 != e_edge.1 {
                graph.add_edge(e_edge.0, e_edge.1, ());
            }

            if !graph.contains_edge(s_edge.0, s_edge.1) && s_edge.0 != s_edge.1 {
                graph.add_edge(s_edge.0, s_edge.1, ());
            }

            if !graph.contains_edge(w_edge.0, w_edge.1) && w_edge.0 != w_edge.1 {
                graph.add_edge(w_edge.0, w_edge.1, ());
            }
        });

        // for i in 0..n {
        //     for j in 0..m {
        //
        //

        //         let i_s1 = i.overflowing_sub(1).1.then(|| n - 1).unwrap_or(i);
        //         let j_s1 = j.overflowing_sub(1).1.then(|| m - 1).unwrap_or(j);

        //         let i_a1 = (i + 1) % n;
        //         let j_a1 = (j + 1) % m;

        //         space.add_connection(xy_to_index(i, j), xy_to_index(i_a1, j)); // n
        //                                                                        // space.add_connection(xy_to_index(i, j), xy_to_index(i_a1, j_a1)); // ne
        //         space.add_connection(xy_to_index(i, j), xy_to_index(i, j_a1)); // e
        //                                                                        // space.add_connection(xy_to_index(i, j), xy_to_index(i_s1, j_a1)); // se
        //         space.add_connection(xy_to_index(i, j), xy_to_index(i_s1, j)); // s
        //                                                                        // space.add_connection(xy_to_index(i, j), xy_to_index(i_s1, j_s1)); // sw
        //         space.add_connection(xy_to_index(i, j), xy_to_index(i, j_s1)); // w
        //                                                                        // space.add_connection(xy_to_index(i, j), xy_to_index(i_a1, j_s1));
        //                                                                        // nw
        //     }
        // }

        // println!("space.connections: {:?}", space.connections.len());
        // println!("space.units: {:?}", sp,ace.units.len());

        graph.edges(nodes[0]).for_each(|edge| {
            println!("edge: {:?}", edge);
        });

        CognitiveSpace {
            rule: Box::new(rule),
            graph,
        }
    }
}

impl<R> CognitiveSpace<R>
where
    R: CognitiveRule + Debug + Clone,
{
    // pub fn step_sync(&self) {
    //     self.connections.iter().for_each(|(from, to)| {
    //         let from_unit = &self.units[*from];
    //         let to_unit = &self.units[*to];

    //         let neighbors = vec![(from_unit.rule.clone(), from_unit.state.clone())];

    //         let _next_state = to_unit.calculate_next_state(neighbors);
    //     });

    //     println!(
    //         "Running cognitive space with rule: {:?}",
    //         self.rule.get_rule_prompt()
    //     );
    // }

    pub fn generate_graph(&self) -> StableGraph<CognitiveUnit, (), Undirected> {
        // let mut nodes = vec![];
        // let mut g = StableUnGraph::with_capacity(self.units.len(), self.connections.len());

        // self.units.iter().for_each(|_unit| {
        //     let node = g.add_node(());
        //     nodes.push(node);
        // });

        // self.connections.iter().for_each(|(from, to)| {
        //     let from = nodes[*from];
        //     let to = nodes[*to];

        //     g.add_edge(from, to, ());
        // });

        self.graph.clone()
    }
}
