use crate::decide::{Choice, DecideError, Decider};

/// Local stand-in used when no checkpoint is loaded.
///
/// Picks the option phrase that occurs most often in the prose. Ties keep the
/// lowest palette index, which is stable for tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct MajorityDecider;

impl Decider for MajorityDecider {
    fn choose(
        &self,
        state: &str,
        _instructions: &str,
        options: &[String],
    ) -> Result<Choice, DecideError> {
        let mut best_index = 0;
        let mut best_count = 0;
        for (index, option) in options.iter().enumerate() {
            let count = occurrences(state, option);
            if count > best_count {
                best_count = count;
                best_index = index;
            }
        }
        let label = options.get(best_index).cloned().unwrap_or_default();
        Ok(Choice {
            label,
            confidence: 1.0,
            probabilities: Vec::new(),
        })
    }
}

/// Whole words only. A one-letter color would otherwise match inside "east".
fn occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| *word == needle)
        .count()
}
