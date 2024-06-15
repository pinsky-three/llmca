use llmca::system::{MessageModelRule, VonNeumannLatticeCognitiveSpace};

fn main() {
    let rule = MessageModelRule::new("Hello, world!".to_string());
    let space = VonNeumannLatticeCognitiveSpace::new_lattice(3, 3, &rule);

    println!("{:?}", space);

    println!("Hello, world!");
}
