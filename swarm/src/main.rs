use std::{path::PathBuf, thread};
use swarm::primitives::CognitiveUnit;

fn main() -> anyhow::Result<()> {
    let model_path: PathBuf = "models/models--HuggingFaceTB--SmolLM2-360M-Instruct-GGUF/snapshots/593b5a2e04c8f3e4ee880263f93e0bd2901ad47f/smollm2-360m-instruct-q8_0.gguf".into();
    let tokenizer_path: PathBuf = "models/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/6849e9f43f1a64e4604f0ef9d23adc8af4b4508f/tokenizer.json".into();

    let mut units = vec![];

    let n = 4;

    for _ in 0..n {
        let device = candle_core::Device::new_metal(0).unwrap();
        let unit = CognitiveUnit::load_model(device, model_path.clone(), tokenizer_path.clone())?;
        units.push(unit);
    }

    let handles = units
        .into_iter()
        .map(|mut unit| {
            // let device_clone = device.clone();
            thread::spawn(move || {
                unit.generate("Hello, how are you?".to_string())
                    .expect("Failed to generate response")
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        let _result = handle.join().expect("Thread panicked");
        // println!("{}", result);
    }

    thread::sleep(std::time::Duration::from_secs(60));

    Ok(())
}
