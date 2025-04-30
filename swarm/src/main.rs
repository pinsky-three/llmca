use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, thread};
use swarm::primitives::{CognitiveUnit, Context, Message};

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

fn main() -> anyhow::Result<()> {
    let model_path: PathBuf = "models/models--HuggingFaceTB--SmolLM2-360M-Instruct-GGUF/snapshots/593b5a2e04c8f3e4ee880263f93e0bd2901ad47f/smollm2-360m-instruct-q8_0.gguf".into();
    let tokenizer_path: PathBuf = "models/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/6849e9f43f1a64e4604f0ef9d23adc8af4b4508f/tokenizer.json".into();

    let mut units = vec![];

    let n = 1;

    for _ in 0..n {
        let device = candle_core::Device::new_metal(0).unwrap();
        let unit = CognitiveUnit::load_model(device, model_path.clone(), tokenizer_path.clone())?;
        units.push(unit);
    }

    let pair_description = CognitiveUnitPair::self_description();

    let system_message = [
        "You're a LLM Cognitive Unit and your unique task is to respond with your next (rule, state) based on your current rule and the states of your neighbors in json format",
        format!("Always respond with a plain json complaint with `CognitiveUnitPair`: {}", pair_description).as_str(),
        "The user pass to you your memory and the neighborhood states as list of 'messages' in json format",
        "Don't put the json in a code block, don't add explanations, just return the json ready to be parsed based on the schema",
        "Only if you rule is empty, you may to propose a new rule and your return it with the response",
        "If you think the rule is wrong, you may to propose a new rule and your return it with the response",
        "Example of valid response: `{\"rule\": \"rule_1\", \"state\": \"state_1\"}`",
    ]
    .join(".\n");

    let handles = units
        .into_iter()
        .map(|mut unit| {
            let system_message = system_message.clone();

            thread::spawn(move || {
                let mut context = Context::new(
                    10,
                    Message {
                        role: "system".to_string(),
                        content: system_message,
                    },
                );

                context.push_message(Message {
                    role: "user".to_string(),
                    content: "you're a pixel in a sunset photo".to_string(),
                });

                for _ in 0.. {
                    // Let's do 3 turns of conversation
                    let result = unit
                        .generate_with_context(&context)
                        .expect("Failed to generate response");

                    context.push_message(result);

                    // println!("{}", result.content);
                    // println!("{:?}", context.messages.last().unwrap().content);

                    for msg in context.messages() {
                        println!("{:?}: {}", msg.role, msg.content);
                    }

                    println!("--------------------------------");

                    // Add a follow-up user message to continue the conversation
                    context.push_message(Message {
                        role: "user".to_string(),
                        content:
                            "Give me a slightly different color (remember only a rgb hex string)"
                                .to_string(),
                    });
                }
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().expect("Thread panicked");
        // println!("{}", result);
    }

    // thread::sleep(std::time::Duration::from_secs(10));

    Ok(())
}
