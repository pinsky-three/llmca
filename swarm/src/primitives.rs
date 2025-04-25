use std::path::PathBuf;

use anyhow::Result;
use candle_core::{
    Device,
    quantized::{ggml_file, gguf_file},
};
use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;
use tokenizers::Tokenizer;

pub struct CognitiveUnit {
    id: String,
    pub model: ModelWeights,
    pub tokenizer: Tokenizer,
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

impl CognitiveUnit {
    pub fn load_model(
        device: &Device,
        model_path: PathBuf,
        tokenizer_path: PathBuf,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(&model_path)?;
        let start = std::time::Instant::now();

        let model = match model_path.extension().and_then(|v| v.to_str()) {
            Some("gguf") => {
                let model =
                    gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
                let mut total_size_in_bytes = 0;
                for (_, tensor) in model.tensor_infos.iter() {
                    let elem_count = tensor.shape.elem_count();
                    total_size_in_bytes +=
                        elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
                }
                println!(
                    "loaded {:?} tensors ({}) in {:.2}s",
                    model.tensor_infos.len(),
                    &format_size(total_size_in_bytes),
                    start.elapsed().as_secs_f32(),
                );
                ModelWeights::from_gguf(model, &mut file, device)?
            }
            Some("ggml" | "bin") | Some(_) | None => {
                let model = ggml_file::Content::read(&mut file, device)
                    .map_err(|e| e.with_path(model_path))?;
                let mut total_size_in_bytes = 0;
                for (_, tensor) in model.tensors.iter() {
                    let elem_count = tensor.shape().elem_count();
                    total_size_in_bytes +=
                        elem_count * tensor.dtype().type_size() / tensor.dtype().block_size();
                }
                println!(
                    "loaded {:?} tensors ({}) in {:.2}s",
                    model.tensors.len(),
                    &format_size(total_size_in_bytes),
                    start.elapsed().as_secs_f32(),
                );
                println!("params: {:?}", model.hparams);
                let default_gqa = 1;

                ModelWeights::from_ggml(model, default_gqa)?
            }
        };

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

        Ok(Self {
            id: "".to_string(),
            model,
            tokenizer,
        })
    }
}
