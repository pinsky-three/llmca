use crate::unit::CognitiveUnit;
use std::{fmt::Debug, marker::PhantomData};

#[derive(Debug, Clone)]
pub struct CognitiveSpace<R>
where
    R: CognitiveRule,
{
    units: Vec<CognitiveUnit>,
    connections: Vec<(usize, usize)>,
    rule: Box<R>,
}

impl<R> CognitiveSpace<R>
where
    R: CognitiveRule + Debug + Clone,
{
    pub fn new(rule: Box<R>) -> Self {
        Self {
            units: vec![],
            connections: vec![],
            rule,
        }
    }

    fn add_unit(&mut self, unit: CognitiveUnit) {
        self.units.push(unit);
    }

    fn add_connection(&mut self, from: usize, to: usize) {
        self.connections.push((from, to));
    }
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
        let mut space = CognitiveSpace::new(Box::new(rule.clone()));

        let xy_to_index = |i: usize, j: usize| -> usize { i * m + j };

        for i in 0..n {
            for j in 0..m {
                let rule = rule.get_rule_prompt();
                let state = vec!["0".to_string()];

                let unit = CognitiveUnit { rule, state };

                space.add_unit(unit);

                let s_i = i;
                let s_j = j;

                let i = i.overflowing_sub(1).1.then(|| n - 1).unwrap_or(i);
                let j = j.overflowing_sub(1).1.then(|| m - 1).unwrap_or(j);

                space.add_connection(xy_to_index(s_i, s_j), xy_to_index(i + 1 % n, j)); // n
                space.add_connection(xy_to_index(s_i, s_j), xy_to_index(i + 1 % n, j + 1 % m)); // ne
                space.add_connection(xy_to_index(s_i, s_j), xy_to_index(i, j + 1 % m)); // e
                space.add_connection(xy_to_index(s_i, s_j), xy_to_index(i - 1 % n, j + 1 % m)); // se
                space.add_connection(xy_to_index(s_i, s_j), xy_to_index(i - 1 % n, j)); // s
                space.add_connection(xy_to_index(s_i, s_j), xy_to_index(i - 1 % n, j - 1 % m)); // sw
                space.add_connection(xy_to_index(s_i, s_j), xy_to_index(i, j - 1 % m)); // w
                space.add_connection(xy_to_index(s_i, s_j), xy_to_index(i + 1 % n, j - 1 % m));
                // nw
            }
        }

        space
    }
}

impl<R> CognitiveSpace<R>
where
    R: CognitiveRule + Debug + Clone,
{
    pub fn step_sync(&self) {
        self.connections.iter().for_each(|(from, to)| {
            let from_unit = &self.units[*from];
            let to_unit = &self.units[*to];

            let neighbors = vec![(from_unit.rule.clone(), from_unit.state.clone())];

            let next_state = to_unit.calculate_next_state(neighbors);
        });

        println!(
            "Running cognitive space with rule: {:?}",
            self.rule.get_rule_prompt()
        );
    }
}
