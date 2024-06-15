use std::{fmt::Debug, marker::PhantomData};

#[derive(Debug, Clone)]
struct CognitiveUnit {
    rule: String,
    state: Vec<String>,
    neighbors: Vec<(String, Vec<String>)>,
}

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
    pub fn new_lattice(n: usize, m: usize, rule: &R) -> CognitiveSpace<R> {
        let mut space = CognitiveSpace::new(Box::new(rule.to_owned()));

        for i in 0..n {
            for j in 0..m {
                let rule = rule.get_rule_prompt();
                let state = vec!["0".to_string()];
                let neighbors = vec![
                    ("n".to_string(), vec!["0".to_string()]),
                    ("ne".to_string(), vec!["0".to_string()]),
                    ("e".to_string(), vec!["0".to_string()]),
                    ("se".to_string(), vec!["0".to_string()]),
                    ("s".to_string(), vec!["0".to_string()]),
                    ("sw".to_string(), vec!["0".to_string()]),
                    ("w".to_string(), vec!["0".to_string()]),
                    ("nw".to_string(), vec!["0".to_string()]),
                ];

                let unit = CognitiveUnit {
                    rule,
                    state,
                    neighbors,
                };

                space.add_unit(unit);
            }
        }

        for i in 0..n {
            for j in 0..m {
                let from = i * m + j;

                // space.units[from]
                //     .clone()
                //     .neighbors
                //     .iter()
                //     .for_each(|(_direction, to)| {
                //         let to = to[0].split('-').collect::<Vec<&str>>();
                //         let to =
                //             to[0].parse::<usize>().unwrap() * m + to[1].parse::<usize>().unwrap();

                //         if to < n * m {
                //             space.add_connection(from, to);
                //         }
                //     });
            }
        }

        space
    }
}
