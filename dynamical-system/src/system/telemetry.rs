use std::time::Duration;

use serde_derive::{Deserialize, Serialize};

use super::unit_next::CognitiveUnitComplex;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct StepTelemetry {
    pub units_total: usize,
    pub units_completed: usize,
    pub resolver_count: usize,
    pub chunks: usize,
    pub llm_failures: usize,
    pub parse_failures: usize,
    pub unique_states: usize,
    pub elapsed_ms: u64,
}

impl StepTelemetry {
    pub fn new(units_total: usize, resolver_count: usize) -> Self {
        Self {
            units_total,
            resolver_count,
            ..Default::default()
        }
    }

    pub fn record_chunk(&mut self) {
        self.chunks += 1;
    }

    pub fn record_unit(&mut self, unit: &CognitiveUnitComplex) {
        self.units_completed += 1;

        if unit.feedback.starts_with("LLM request failed") {
            self.llm_failures += 1;
        } else if unit.feedback.starts_with("Structured output failed") || !unit.feedback.is_empty()
        {
            self.parse_failures += 1;
        }
    }

    pub fn finish(&mut self, elapsed: Duration, unique_states: usize) {
        self.elapsed_ms = elapsed.as_millis().try_into().unwrap_or(u64::MAX);
        self.unique_states = unique_states;
    }
}
