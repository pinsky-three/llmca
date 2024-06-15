use dotenv::dotenv;
use llmca::system::{MessageModelRule, VonNeumannLatticeCognitiveSpace};

fn main() {
    dotenv().ok();

    let rule = MessageModelRule::new("Hello, world!".to_string());
    let space = VonNeumannLatticeCognitiveSpace::new_lattice(3, 3, rule);

    println!("{:?}", space);

    println!("Hello, world!");
}
