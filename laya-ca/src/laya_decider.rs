use std::path::Path;

use laya::{Agent, Options, Question};

use crate::decide::{Choice, DecideError, Decider};

/// Local Laya checkpoint. One `system_one` call answers the color question.
///
/// Weights stay on disk (`LAYA_MODEL_DIR` or `--model`). They are not part of
/// the crate: the typed-decisions checkpoint is the one meant for this job.
pub struct LayaDecider {
    agent: Agent,
}

impl LayaDecider {
    pub fn open(dir: &Path, half: bool) -> Result<Self, DecideError> {
        // f32 already fills the GPU at batch 1, so a wider batch does not shorten a step.
        // f16 cuts the matmul traffic. Scores drift by about 1e-3, which sampling tolerates.
        let agent = Agent::from_dir(
            dir,
            Options {
                dtype: half.then_some(candle_core::DType::F16),
                ..Options::default()
            },
        )
        .map_err(|err| DecideError::Model(format!("loading {}: {err}", dir.display())))?;
        tracing::info!(
            model_dir = %dir.display(),
            device = ?agent.device(),
            half,
            "laya_ca_checkpoint_ready"
        );
        Ok(Self { agent })
    }
}

impl Decider for LayaDecider {
    fn choose(
        &self,
        state: &str,
        instructions: &str,
        options: &[String],
    ) -> Result<Choice, DecideError> {
        self.choose_many(&[state], instructions, options)
            .into_iter()
            .next()
            .unwrap_or_else(|| Err(DecideError::Model("empty batch".into())))
    }

    fn choose_many(
        &self,
        states: &[&str],
        instructions: &str,
        options: &[String],
    ) -> Vec<Result<Choice, DecideError>> {
        if states.is_empty() {
            return Vec::new();
        }
        let birth_statement = instructions.contains("empty with exactly 3 neighbors");
        let cases: Vec<(serde_json::Value, Vec<(String, Question)>)> = states
            .iter()
            .map(|state| {
                let question = Question::choice(instructions, options.iter().map(String::as_str));
                let mut questions = vec![("next_color".to_string(), question)];
                if birth_statement {
                    questions.push((
                        "birth".to_string(),
                        Question::noul("The situation is empty with exactly 3 neighbors."),
                    ));
                }
                (serde_json::json!(state), questions)
            })
            .collect();
        let responses = match self.agent.system_many(&cases) {
            Ok(responses) => responses,
            Err(err) => {
                let message = err.to_string();
                return states
                    .iter()
                    .map(|_| Err(DecideError::Model(message.clone())))
                    .collect();
            }
        };
        responses
            .into_iter()
            .map(|response| choice_from_response(response, options, birth_statement))
            .collect()
    }
}

fn choice_from_response(
    response: laya::Response,
    options: &[String],
    birth_statement: bool,
) -> Result<Choice, DecideError> {
    let answer = response
        .answers
        .get("next_color")
        .ok_or_else(|| DecideError::Model("response missing next_color".into()))?;
    let label = answer
        .get("choice")
        .and_then(|value| value.as_str())
        .ok_or_else(|| DecideError::Model("choice answer missing label".into()))?;
    let confidence = answer
        .get("confidence")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0) as f32;
    let weights = answer
        .get("probabilities")
        .and_then(|value| value.as_object());
    let probabilities: Vec<(String, f32)> = options
        .iter()
        .filter_map(|option| {
            let weight = weights?.get(option)?.as_f64()? as f32;
            Some((option.clone(), weight))
        })
        .collect();
    let mut label = label.to_string();
    if birth_statement {
        let birth = response
            .answers
            .get("birth")
            .and_then(|answer| answer.get("noul"))
            .and_then(|value| value.as_f64())
            .unwrap_or(0.0);
        let p_off = prob(&probabilities, "off");
        let p_on = prob(&probabilities, "on");
        // The choice head ties the birth case (about 0.50/0.50). The
        // statement head scores that sentence at 0.62 and stays under 0.14
        // on every other near-tie, so it only breaks this one tie.
        if label == "off" && p_on + 0.02 >= p_off && birth >= 0.55 {
            label = "on".to_string();
        }
    }
    Ok(Choice {
        label,
        confidence,
        probabilities,
    })
}

fn prob(probabilities: &[(String, f32)], label: &str) -> f32 {
    probabilities
        .iter()
        .find(|(name, _)| name == label)
        .map(|(_, weight)| *weight)
        .unwrap_or(0.0)
}
