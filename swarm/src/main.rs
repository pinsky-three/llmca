// use clap::{Parser, ValueEnum};
use rand::Rng;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use candle_core::Tensor;
use candle_core::quantized::{ggml_file, gguf_file};
use candle_transformers::generation::{LogitsProcessor, Sampling};

// use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;

#[derive(Debug)]
enum Prompt {
    // Interactive,
    Chat,
    // One(String),
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

fn main() -> anyhow::Result<()> {
    // use tracing_chrome::ChromeLayerBuilder;
    // use tracing_subscriber::prelude::*;

    // let args = Args::parse();

    // #[cfg(feature = "cuda")]
    // candle::quantized::cuda::set_force_dmmv(args.force_dmmv);

    // candle_core::cuda::set_gemm_reduced_precision_f16(true);
    // candle_core::cuda::set_gemm_reduced_precision_bf16(true);

    // let _guard = if args.tracing {
    //     let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    //     tracing_subscriber::registry().with(chrome_layer).init();
    //     Some(guard)
    // } else {
    //     None
    // };

    // println!(
    //     "avx: {}, neon: {}, simd128: {}, f16c: {}",
    //     candle_core::utils::with_avx(),
    //     candle_core::utils::with_neon(),
    //     candle_core::utils::with_simd128(),
    //     candle_core::utils::with_f16c()
    // );
    // println!(
    //     "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
    //     args.temperature, args.repeat_penalty, args.repeat_last_n
    // );

    let model_path: PathBuf = "models/models--HuggingFaceTB--SmolLM2-360M-Instruct-GGUF/snapshots/593b5a2e04c8f3e4ee880263f93e0bd2901ad47f/smollm2-360m-instruct-q8_0.gguf".into();
    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();
    // let device = candle_examples::device(false)?;

    let device = candle_core::Device::new_metal(0).unwrap();

    let mut model = match model_path.extension().and_then(|v| v.to_str()) {
        Some("gguf") => {
            let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
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
            ModelWeights::from_gguf(model, &mut file, &device)?
        }
        Some("ggml" | "bin") | Some(_) | None => {
            let model = ggml_file::Content::read(&mut file, &device)
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
    println!("model built");

    // let tokenizer_path = {
    //     let api = hf_hub::api::sync::Api::new()?;
    //     let repo = "HuggingFaceTB/SmolLM2-360M-Instruct";
    //     let api = api.model(repo.to_string());
    //     // println!("api: {:?}", api);
    //     api.get("tokenizer.json")?
    // };

    let tokenizer_path: PathBuf = "models/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/6849e9f43f1a64e4604f0ef9d23adc8af4b4508f/tokenizer.json".into();

    println!("tokenizer_path: {:?}", tokenizer_path);

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    // let tokenizer = args.tokenizer()?;
    let mut tos = TokenOutputStream::new(tokenizer);
    // let prompt = match args.prompt.as_deref() {
    //     Some("chat") => Prompt::Chat,
    //     Some("interactive") => Prompt::Interactive,
    //     Some(s) => Prompt::One(s.to_string()),
    //     None => Prompt::One(DEFAULT_PROMPT.to_string()),
    // };

    let prompt = Prompt::Chat; //("tell me a joke".to_string());

    let mut pre_prompt_tokens = vec![];
    for prompt_index in 0.. {
        let prompt_str = match &prompt {
            // Prompt::One(prompt) => prompt.clone(),
            Prompt::Chat => {
                // let is_interactive = matches!(prompt, Prompt::Interactive);
                print!("> ");
                std::io::stdout().flush()?;
                let mut prompt = String::new();
                std::io::stdin().read_line(&mut prompt)?;
                if prompt.ends_with('\n') {
                    prompt.pop();
                    if prompt.ends_with('\r') {
                        prompt.pop();
                    }
                }
                // if args.which.is_open_chat() {
                //     format!("GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:")
                // } else if args.which.is_zephyr() {
                //     if prompt_index == 0 || is_interactive {
                //         format!("<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>",)
                //     } else {
                //         format!("<|user|>\n{prompt}</s>\n<|assistant|>")
                //     }
                // } else if args.which.is_mistral() {
                //     format!("[INST] {prompt} [/INST]")
                // } else if args.which.is_deepseek() {
                //     format!("<｜User｜>{prompt}<｜Assistant｜>")
                // } else {
                //     prompt
                // }
                prompt
            }
        };
        print!("{}", &prompt_str);
        let tokens = tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;
        // if args.verbose_prompt {
        //     for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
        //         let token = token.replace('▁', " ").replace("<0x0A>", "\n");
        //         println!("{id:7} -> '{token}'");
        //     }
        // }

        let sample_len = 512usize;
        let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
        let to_sample = sample_len.saturating_sub(1);
        let prompt_tokens = if prompt_tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };

        let mut rng = rand::rng();

        let mut all_tokens = vec![];
        let mut logits_processor = {
            let temperature = 0.7;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                // match (args.top_k, args.top_p) {
                //     (None, None) => Sampling::All { temperature },
                //     (Some(k), None) => Sampling::TopK { k, temperature },
                //     (None, Some(p)) => Sampling::TopP { p, temperature },
                //     (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                // }
                Sampling::All { temperature }
            };
            LogitsProcessor::from_sampling(rng.random::<u64>(), sampling)
        };

        let start_prompt_processing = std::time::Instant::now();
        // let mut next_token = if !args.split_prompt {
        //     let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
        //     let logits = model.forward(&input, 0)?;
        //     let logits = logits.squeeze(0)?;
        //     logits_processor.sample(&logits)?
        // } else {
        //     let mut next_token = 0;
        //     for (pos, token) in prompt_tokens.iter().enumerate() {
        //         let input = Tensor::new(&[*token], &device)?.unsqueeze(0)?;
        //         let logits = model.forward(&input, pos)?;
        //         let logits = logits.squeeze(0)?;
        //         next_token = logits_processor.sample(&logits)?
        //     }
        //     next_token
        // };
        let mut next_token = {
            let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };

        let prompt_dt = start_prompt_processing.elapsed();

        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        let eos_token = "<|endoftext|>";

        let repeat_penalty = 1.1;
        let repeat_last_n = 64;

        let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
        let start_post_prompt = std::time::Instant::now();
        let mut sampled = 0;
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, prompt_tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            let logits = if repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = tos.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            sampled += 1;
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        let dt = start_post_prompt.elapsed();
        println!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            prompt_tokens.len(),
            prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        println!(
            "{sampled:4} tokens generated: {:.2} token/s",
            sampled as f64 / dt.as_secs_f64(),
        );

        match prompt {
            // Prompt::One(_) => break,
            // Prompt::Interactive => {}
            Prompt::Chat => {
                pre_prompt_tokens = [prompt_tokens.as_slice(), all_tokens.as_slice()].concat()
            }
        }
    }

    Ok(())
}

use candle_core::Result;

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle_core::bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}
