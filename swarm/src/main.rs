use std::{path::PathBuf, thread};
use swarm::primitives::{CognitiveUnit, Context, Message};

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

    let handles = units
        .into_iter()
        .map(|mut unit| {
            thread::spawn(move || {
                let mut context = Context::new(
                    10,
                    Message {
                        role: "system".to_string(),
                        content: "You only answer in hex rgb code".to_string(),
                    },
                );

                context.push_message(Message {
                    role: "user".to_string(),
                    content: "you're a pixel in a sunset photo, give me the #rrggbb code"
                        .to_string(),
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
