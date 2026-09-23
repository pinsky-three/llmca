//! Rust inference for [Laya](https://huggingface.co/convaiinnovations/laya), a non-autoregressive
//! typed-decision model: a ModernBERT-large encoder plus an RL-trained decision head.
//!
//! Give it a state (text or JSON) and a set of typed questions; it returns typed answers with
//! calibrated probabilities in a single forward pass. It never generates text.
//!
//! ```no_run
//! use laya::{Agent, Question};
//! use serde_json::json;
//!
//! let agent = Agent::from_dir("models/laya-base", Default::default())?;
//! let answers = agent.system_one(
//!     &json!("My card was charged twice for the same order."),
//!     &[("department".into(), Question::choice("Which team owns this?", ["billing", "technical", "sales"]))]
//!         .into_iter()
//!         .collect(),
//! )?;
//! println!("{}", serde_json::to_string_pretty(&answers)?);
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod config;
pub mod model;
pub mod modernbert;
pub mod question;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

pub use config::{AgentConfig, EncoderConfig};
pub use question::{QType, Question};

use question::SpecialIds;

/// How to place and run the model.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// `None` picks the best available: CUDA, then Metal, then CPU.
    pub device: Option<Device>,
    /// Currently f32 only; see the note on [`Agent::from_dir`].
    pub dtype: Option<DType>,
}

/// One typed answer. The variant matches the question's type.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice {
        /// The argmax option.
        choice: String,
        probabilities: Map<String, Value>,
        /// 1 - normalized entropy of the answer distribution.
        confidence: Value,
        rl_agent: Meta,
    },
    Score {
        /// Expectation over level indices.
        score: Value,
        legend: Map<String, Value>,
        probabilities: Map<String, Value>,
        confidence: Value,
        rl_agent: Meta,
    },
    Noul {
        /// P(the statement holds).
        noul: Value,
        rl_agent: Meta,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct Meta {
    /// The act head's probability of answering rather than escalating.
    pub act_probability: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub model: String,
    pub answers: Map<String, Value>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Which part of the rendered sequence a token belongs to.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Segment {
    /// `[CLS]` / `[SEP]` structure.
    Special,
    /// `"<type> question: <instructions>"`.
    Head,
    /// An option's text. `option` carries its index.
    Option,
    /// The serialized state.
    State,
}

/// One token of the rendered sequence.
#[derive(Debug, Clone, Serialize)]
pub struct TokenView {
    /// The token text, with the byte-level BPE space marker turned back into a space.
    pub text: String,
    pub id: u32,
    pub segment: Segment,
    /// True for the `[MASK]` marker the head reads this option's logit from.
    pub marker: bool,
    /// The option this token belongs to, for `Option` and marker tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub option: Option<usize>,
}

/// The exact sequence the encoder sees for one question, for inspection and debugging.
#[derive(Debug, Clone, Serialize)]
pub struct PromptView {
    pub tokens: Vec<TokenView>,
    /// Rendered option texts, in label-index order.
    pub options: Vec<String>,
    /// Marker positions in the sequence.
    pub markers: Vec<usize>,
    pub total_tokens: usize,
    pub max_len: usize,
    /// True when the state was cut to fit `max_len`.
    pub state_truncated: bool,
}

pub struct Agent {
    model: model::DecisionModel,
    tokenizer: Tokenizer,
    specials: SpecialIds,
    pub cfg: AgentConfig,
}

impl Agent {
    /// Load a checkpoint directory containing `model.safetensors`, `encoder/`, `tokenizer/`
    /// and `rl_agent_config.json`.
    pub fn from_dir(dir: impl AsRef<Path>, opts: Options) -> Result<Self> {
        let dir = dir.as_ref();
        let device = match opts.device {
            Some(d) => d,
            None => default_device(),
        };
        // Weights are f16 on disk. The vendored encoder builds its attention masks in the
        // model's dtype (upstream candle hardcodes f32), so an f16 backbone now loads and
        // runs on an accelerator, halving weight memory from 1.69 GB to 0.84 GB.
        //
        // f32 stays the default. f16 costs bit-exact agreement between a batched answer and
        // the same question asked alone: they drift by around 1e-3, which is immaterial to a
        // thresholded decision and visible if you compare payloads. Ask for it explicitly
        // when the memory matters more, which is the usual case when this model is resident
        // alongside a larger one.
        let dtype = opts.dtype.unwrap_or(DType::F32);
        if dtype == DType::F16 && matches!(device, Device::Cpu) {
            anyhow::bail!("f16 needs an accelerator: candle's CPU kernels have no f16 path");
        }

        let cfg = AgentConfig::load(dir.join("rl_agent_config.json"))?;
        let enc_cfg = EncoderConfig::load(dir.join("encoder/config.json"))?;

        let tok_path = dir.join("tokenizer/tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer {}: {e}", tok_path.display()))?;
        let specials = resolve_specials(&tokenizer)?;

        let weights = dir.join("model.safetensors");
        let tensors = candle_core::safetensors::load(&weights, &device)
            .with_context(|| format!("loading weights {}", weights.display()))?;
        // The checkpoint stores the backbone under `encoder.*`; candle's ModernBert expects `model.*`.
        let tensors: HashMap<String, Tensor> = tensors
            .into_iter()
            .map(|(k, v)| match k.strip_prefix("encoder.") {
                Some(rest) => (format!("model.{rest}"), v),
                None => (k, v),
            })
            .collect();
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);

        let head_layers = if cfg.head_layers == 0 {
            2
        } else {
            cfg.head_layers
        };
        let model = model::DecisionModel::load(vb, &enc_cfg, head_layers, 2, device, dtype)?;

        Ok(Self {
            model,
            tokenizer,
            specials,
            cfg,
        })
    }

    /// Answer every question about `state` in a single batched forward pass.
    pub fn system_one(
        &self,
        state: &Value,
        questions: &Vec<(String, Question)>,
    ) -> Result<Response> {
        let mut responses = self.system_many(&[(state.clone(), questions.clone())])?;
        Ok(responses.pop().unwrap_or(Response {
            model: self.model_name(),
            answers: Map::new(),
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
        }))
    }

    /// One forward for many states. Each case keeps its own state text, so a
    /// cellular automaton can score a row of cells together. Rows do not attend
    /// to each other.
    pub fn system_many(
        &self,
        cases: &[(Value, Vec<(String, Question)>)],
    ) -> Result<Vec<Response>> {
        struct Row<'a> {
            case_idx: usize,
            qid: &'a str,
            question: &'a Question,
        }

        let mut rows: Vec<Row<'_>> = Vec::new();
        let mut encoded = Vec::new();
        for (case_idx, (state, questions)) in cases.iter().enumerate() {
            for (qid, question) in questions {
                let sequence = question::build_sequence(
                    &self.tokenizer,
                    &self.specials,
                    state,
                    question,
                    self.cfg.max_len,
                    self.cfg.head_max_len,
                )
                .with_context(|| format!("question {qid:?}"))?;
                encoded.push(sequence);
                rows.push(Row {
                    case_idx,
                    qid,
                    question,
                });
            }
        }

        let mut responses = cases
            .iter()
            .map(|_| Response {
                model: self.model_name(),
                answers: Map::new(),
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                },
            })
            .collect::<Vec<_>>();
        if rows.is_empty() {
            return Ok(responses);
        }

        let n = encoded.len();
        let l = encoded.iter().map(|e| e.ids.len()).max().unwrap();
        let kmax = encoded.iter().map(|e| e.markers.len()).max().unwrap();

        let mut ids = vec![self.specials.pad; n * l];
        let mut att = vec![0u32; n * l];
        let mut mpos = vec![0u32; n * kmax];
        let mut mmask = vec![vec![false; kmax]; n];
        let mut qtypes = Vec::with_capacity(n);
        let mut input_tokens = 0usize;

        for (i, (sequence, row)) in encoded.iter().zip(&rows).enumerate() {
            ids[i * l..i * l + sequence.ids.len()].copy_from_slice(&sequence.ids);
            for j in 0..sequence.ids.len() {
                att[i * l + j] = 1;
            }
            input_tokens += sequence.ids.len();
            for (j, marker) in sequence.markers.iter().enumerate() {
                mpos[i * kmax + j] = *marker as u32;
                mmask[i][j] = true;
            }
            qtypes.push(row.question.qtype.index() as u32);
        }

        let dev = &self.model.device;
        let input_ids = Tensor::from_vec(ids, (n, l), dev)?;
        let attention_mask = Tensor::from_vec(att, (n, l), dev)?;
        let marker_pos = Tensor::from_vec(mpos, (n, kmax), dev)?;
        let qtype = Tensor::from_vec(qtypes, n, dev)?;

        let out = self
            .model
            .forward(&input_ids, &attention_mask, &marker_pos, &mmask, &qtype)?;

        for (r, (row, sequence)) in rows.iter().zip(&encoded).enumerate() {
            let k = sequence.markers.len();
            let t = self.cfg.temperature_for(row.question.qtype.index(), k);
            let z: Vec<f32> = out.logits[r][..k].iter().map(|v| v / t).collect();
            let p = model::stable_softmax(&z);
            let meta = Meta {
                act_probability: json_f32(round4(&out.act[r][0])),
            };
            let answer = match row.question.qtype {
                QType::Choice => {
                    let keys = row.question.choice_keys()?;
                    let best = argmax(&p);
                    Answer::Choice {
                        choice: keys[best].clone(),
                        probabilities: keys
                            .iter()
                            .cloned()
                            .zip(p.iter().map(|v| json_f32(round4(v))))
                            .collect(),
                        confidence: json_f32(round4(&confidence_from_probs(&p, k))),
                        rl_agent: meta,
                    }
                }
                QType::Score => {
                    let score: f32 = p.iter().enumerate().map(|(i, v)| i as f32 * v).sum();
                    Answer::Score {
                        score: json_f32(round4(&score)),
                        legend: row
                            .question
                            .score_levels()
                            .into_iter()
                            .enumerate()
                            .map(|(i, c)| (i.to_string(), Value::String(c)))
                            .collect(),
                        probabilities: p
                            .iter()
                            .enumerate()
                            .map(|(i, v)| (i.to_string(), json_f32(round4(v))))
                            .collect(),
                        confidence: json_f32(round4(&confidence_from_probs(&p, k))),
                        rl_agent: meta,
                    }
                }
                QType::Noul => Answer::Noul {
                    noul: json_f32(round4(&p[1])),
                    rl_agent: meta,
                },
            };
            responses[row.case_idx]
                .answers
                .insert(row.qid.to_string(), serde_json::to_value(answer)?);
            responses[row.case_idx].usage.input_tokens += sequence.ids.len();
        }
        let _ = input_tokens;
        Ok(responses)
    }

    /// The exact token sequence for one question, annotated by segment. Useful for showing
    /// where the option markers land and whether the state was truncated.
    pub fn render_prompt(&self, state: &Value, q: &Question) -> Result<PromptView> {
        let e = question::build_sequence(
            &self.tokenizer,
            &self.specials,
            state,
            q,
            self.cfg.max_len,
            self.cfg.head_max_len,
        )?;
        let options = q.render_options()?;

        let mut tokens = Vec::with_capacity(e.ids.len());
        for (i, id) in e.ids.iter().enumerate() {
            let text = self
                .tokenizer
                .id_to_token(*id)
                .unwrap_or_else(|| format!("<{id}>"))
                .replace('\u{0120}', " ")
                .replace('\u{010a}', "\\n");
            let marker = e.markers.binary_search(&i).is_ok();
            // [CLS] head [SEP] (marker opt)* [SEP] state [SEP]
            let segment =
                if marker || i == 0 || i == e.head_sep || i == e.opts_sep || i + 1 == e.ids.len() {
                    Segment::Special
                } else if i < e.head_sep {
                    Segment::Head
                } else if i < e.opts_sep {
                    Segment::Option
                } else {
                    Segment::State
                };
            // Markers and option text both belong to the option whose marker most recently opened.
            let option = if i >= *e.markers.first().unwrap_or(&usize::MAX) && i < e.opts_sep {
                e.markers.iter().rposition(|m| *m <= i)
            } else {
                None
            };
            tokens.push(TokenView {
                text,
                id: *id,
                segment,
                marker,
                option,
            });
        }

        // The state fits when the sequence came in under budget.
        let state_truncated = e.ids.len() >= self.cfg.max_len;

        Ok(PromptView {
            total_tokens: e.ids.len(),
            tokens,
            options,
            markers: e.markers,
            max_len: self.cfg.max_len,
            state_truncated,
        })
    }

    /// The device the model is resident on.
    pub fn device(&self) -> &Device {
        &self.model.device
    }

    fn model_name(&self) -> String {
        if self.cfg.model_name.is_empty() {
            "rl-agent".to_string()
        } else {
            self.cfg.model_name.clone()
        }
    }
}

fn default_device() -> Device {
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    Device::Cpu
}

fn resolve_specials(tok: &Tokenizer) -> Result<SpecialIds> {
    let id = |t: &str| -> Result<u32> {
        tok.token_to_id(t)
            .ok_or_else(|| anyhow::anyhow!("tokenizer has no {t} token"))
    };
    Ok(SpecialIds {
        cls: id("[CLS]")?,
        sep: id("[SEP]")?,
        mask: id("[MASK]")?,
        pad: id("[PAD]")?,
        mask_text: "[MASK]".to_string(),
    })
}

fn argmax(p: &[f32]) -> usize {
    p.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Serialize an f32 without the noise of its f64 widening (0.9641 rather than 0.9641000032424927).
fn json_f32(v: f32) -> Value {
    serde_json::Number::from_f64(format!("{v}").parse::<f64>().unwrap_or(v as f64))
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn round4(v: &f32) -> f32 {
    (v * 10_000.0).round() / 10_000.0
}

/// 1 - normalized entropy of the answer distribution.
fn confidence_from_probs(p: &[f32], k: usize) -> f32 {
    if k < 2 {
        return 1.0;
    }
    let ent: f32 = -p[..k]
        .iter()
        .map(|v| v * v.clamp(1e-12, 1.0).ln())
        .sum::<f32>();
    1.0 - ent / (k as f32).ln()
}
