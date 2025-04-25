use std::{io::Write, path::PathBuf};

use anyhow::Result;
use candle_core::{
    Device, Tensor,
    quantized::{ggml_file, gguf_file},
};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_llama as model,
};
use model::ModelWeights;
use rand::Rng;
use serde::Serialize;
// use tinytemplate::TinyTemplate;
use tokenizers::Tokenizer;

pub struct CognitiveUnit {
    _id: String,
    pub device: Device,
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
        device: Device,
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

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

        Ok(Self {
            _id: "".to_string(),
            model,
            tokenizer,
            device,
        })
    }
    pub fn generate(&mut self, prompt: String) -> Result<String> {
        let mut tos = TokenOutputStream::new(self.tokenizer.clone());

        // let prompt = Prompt::Chat; //("tell me a joke".to_string());

        // let mut pre_prompt_tokens = vec![];

        // let prompt_str = match &prompt {
        //     // Prompt::One(prompt) => prompt.clone(),
        //     Prompt::Chat => {
        //         // let is_interactive = matches!(prompt, Prompt::Interactive);
        //         print!("> ");
        //         std::io::stdout().flush()?;
        //         let mut prompt = String::new();
        //         std::io::stdin().read_line(&mut prompt)?;
        //         if prompt.ends_with('\n') {
        //             prompt.pop();
        //             if prompt.ends_with('\r') {
        //                 prompt.pop();
        //             }
        //         }

        //         prompt
        //     }
        // };
        let prompt_str = prompt;

        // print!("{}", prompt_str);

        let tokens = tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;

        let sample_len = 1024_usize;
        // let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
        let prompt_tokens = tokens.get_ids();
        let to_sample = sample_len.saturating_sub(1);
        let prompt_tokens = if prompt_tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens.to_vec()
        };

        let mut rng = rand::rng();

        let mut all_tokens = vec![];
        let mut all_tokens_str = String::new();

        let mut logits_processor = {
            let temperature = 0.2;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                // Sampling::All { temperature }
                Sampling::TopP {
                    p: 0.9,
                    temperature,
                }
            };
            LogitsProcessor::from_sampling(rng.random::<u64>(), sampling)
        };

        // let start_prompt_processing = std::time::Instant::now();

        let mut next_token = {
            let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };

        // let prompt_dt = start_prompt_processing.elapsed();

        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            // print!("{t}");
            all_tokens_str.push_str(&t);
            std::io::stdout().flush()?;
        }

        let eos_token = "<|im_end|>";

        let repeat_penalty = 1.1;
        let repeat_last_n = 64;

        let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
        // let start_post_prompt = std::time::Instant::now();
        // let mut sampled = 0;
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, prompt_tokens.len() + index)?;
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
                // print!("{t}");
                all_tokens_str.push_str(&t);
                std::io::stdout().flush()?;
            }
            // sampled += 1;
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
            // print!("{rest}");
            all_tokens_str.push_str(&rest);
        }

        std::io::stdout().flush()?;
        // let dt = start_post_prompt.elapsed();
        // println!(
        //     "\n\n{:4} prompt tokens processed: {:.2} token/s",
        //     prompt_tokens.len(),
        //     prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
        // );
        // println!(
        //     "{sampled:4} tokens generated: {:.2} token/s",
        //     sampled as f64 / dt.as_secs_f64(),
        // );

        let _ = [prompt_tokens.as_slice(), all_tokens.as_slice()].concat();

        Ok(all_tokens_str)
    }

    pub fn generate_with_context(&mut self, context: &Context) -> Result<Message> {
        let prompt = context.compile();

        let message = self.generate(prompt)?;

        Ok(Message::from_string(message))
    }
}

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

    fn decode(&self, tokens: &[u32]) -> candle_core::Result<String> {
        match self.tokenizer.decode(tokens, false) {
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

    pub fn decode_all(&self) -> candle_core::Result<String> {
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

#[derive(Serialize, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: String, content: String) -> Self {
        Self { role, content }
    }

    pub fn from_string(s: String) -> Self {
        let parts = s.splitn(2, '\n').collect::<Vec<&str>>();
        if parts.len() == 2 {
            let mut role = parts[0].trim().to_string();
            // Remove <|im_start|> prefix from role if present
            if role.starts_with("<|im_start|>") {
                role = role.trim_start_matches("<|im_start|>").trim().to_string();
            }

            let mut content = parts[1].trim().to_string();
            // Remove trailing <|im_end|> if present
            if content.ends_with("<|im_end|>") {
                content.truncate(content.len() - "<|im_end|>".len());
                content = content.trim_end().to_string(); // Trim potential whitespace before the tag
            }
            Self { role, content }
        } else {
            // If split fails, assume the whole string is content and clean it
            let mut content = s.trim().to_string();
            if content.ends_with("<|im_end|>") {
                content.truncate(content.len() - "<|im_end|>".len());
                content = content.trim_end().to_string();
            }
            // Assign a default role since we couldn't parse one
            // Also check for <|im_start|> prefix here just in case, though unlikely
            let mut role = "unknown".to_string();
            if s.trim().starts_with("<|im_start|>") {
                // Attempt to extract role if possible, might indicate malformed input
                if let Some(newline_pos) = s.trim().find('\n') {
                    let potential_role = s.trim()[..newline_pos]
                        .trim_start_matches("<|im_start|>")
                        .trim();
                    if !potential_role.is_empty() {
                        role = potential_role.to_string();
                    }
                }
            }
            Self { role, content }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Context {
    size: usize,
    pub messages: Vec<Message>,
}

impl Context {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            messages: Vec::new(),
        }
    }

    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self {
            size: messages.len() * 2,
            messages,
        }
    }

    pub fn add_message(&mut self, message: Message) {
        if (self.messages.len() - 1) > self.size {
            self.messages
                .drain(2..self.size)
                .collect::<Vec<_>>()
                .push(message);
        } else {
            self.messages.push(message);
        }
    }

    pub fn compile(&self) -> String {
        let mut prompt = String::new();
        for message in &self.messages {
            prompt += &format!(
                "<|im_start|>{}\n{}\n<|im_end|>\n",
                message.role, message.content
            );
        }
        prompt
    }

    // New function to parse the output
    // pub fn parse_output_to_messages(output: &str) -> Vec<Message> {
    //     let mut messages = Vec::new();
    //     let parts = output.split("<|im_start|>");

    //     for part in parts {
    //         let trimmed_part = part.trim();
    //         if trimmed_part.is_empty() {
    //             continue;
    //         }

    //         if let Some(end_pos) = trimmed_part.find("<|im_end|>") {
    //             let content_part = &trimmed_part[..end_pos].trim();
    //             if let Some(newline_pos) = content_part.find('\n') {
    //                 let role = content_part[..newline_pos].trim().to_string();
    //                 let content = content_part[newline_pos + 1..].trim().to_string();
    //                 messages.push(Message { role, content });
    //             }
    //         }
    //     }
    //     messages
    // }
}

impl Default for Context {
    fn default() -> Self {
        Self::new(10)
    }
}
