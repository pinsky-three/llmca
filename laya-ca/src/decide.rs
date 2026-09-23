/// Ask what is true of this pixel, not which policy bucket it falls into.
/// Laya has not read a rule string; it only scores the phrases below.
pub const NEXT_COLOR_INSTRUCTIONS: &str = "Next color?";

#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub label: String,
    pub confidence: f32,
    /// Option weights in palette order. Empty means "use `label`" (the argmax).
    pub probabilities: Vec<(String, f32)>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecideError {
    #[error("laya decision failed: {0}")]
    Model(String),
}

/// One typed decision about a prose state.
///
/// The lattice never sees tokens or logits. Adapters (Laya, the majority
/// stand-in) both return a label from `options`.
pub trait Decider {
    fn choose(
        &self,
        state: &str,
        instructions: &str,
        options: &[String],
    ) -> Result<Choice, DecideError>;

    /// Score several states that share one question. The default asks them one by one.
    fn choose_many(
        &self,
        states: &[&str],
        instructions: &str,
        options: &[String],
    ) -> Vec<Result<Choice, DecideError>> {
        states
            .iter()
            .map(|state| self.choose(state, instructions, options))
            .collect()
    }
}

/// Which sentence and option list a generation uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Prompt {
    /// Sunset prose and the full 27-color cube.
    #[default]
    Sunset,
    /// Conway's rule, dead `#000000` and live `#ffffff`.
    Life,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepConfig {
    /// Below this, the pixel keeps its color. Weak calls should not flicker.
    pub min_confidence: f32,
    pub instructions: String,
    /// `1` keeps the argmax. Higher values flatten the distribution, then sample.
    /// The 27-color bucket is shipped at temperature ~0.1, so the raw argmax locks.
    pub temperature: f32,
    pub prompt: Prompt,
    /// Cells per forward pass. `1` keeps one call per cell.
    pub batch: usize,
}

impl Default for StepConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.0,
            instructions: NEXT_COLOR_INSTRUCTIONS.to_string(),
            temperature: 4.0,
            prompt: Prompt::Sunset,
            batch: 1,
        }
    }
}
