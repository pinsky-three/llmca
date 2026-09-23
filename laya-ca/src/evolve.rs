use std::time::Instant;

use crate::decide::{Choice, Decider, Prompt, StepConfig};
use crate::lattice::Lattice;
use crate::life;
use crate::palette;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StepReport {
    pub cells: usize,
    pub rewrites: usize,
    pub held: usize,
    pub unchanged: usize,
    pub failures: usize,
    pub unique_swatches: usize,
    pub mean_confidence: f32,
    pub elapsed_ms: u64,
}

/// One synchronous generation. Each cell is one decision; a failure keeps the
/// previous color so a bad call cannot punch a hole in the picture.
pub fn evolve(
    lattice: &Lattice,
    decider: &dyn Decider,
    cfg: &StepConfig,
    rng: &mut u64,
) -> (Lattice, StepReport) {
    let started = Instant::now();
    // The palette is the cell's state space. The neighborhood is only the input,
    // so the next color may be the one it already has.
    let options: Vec<String> = match cfg.prompt {
        Prompt::Sunset => palette::colors()
            .iter()
            .map(|swatch| swatch.label.to_string())
            .collect(),
        Prompt::Life => life::options(),
    };
    let mut next = lattice.clone();
    let mut rewrites = 0;
    let mut held = 0;
    let mut unchanged = 0;
    let mut failures = 0;
    let mut confidence_sum = 0.0;
    let mut confidence_n = 0usize;

    let jobs: Vec<(usize, usize, String)> = (0..lattice.height())
        .flat_map(|y| {
            (0..lattice.width()).map(move |x| {
                let state = match cfg.prompt {
                    Prompt::Sunset => lattice.describe(x, y),
                    Prompt::Life => life::describe(lattice, x, y),
                };
                (x, y, state)
            })
        })
        .collect();
    let batch = cfg.batch.max(1);
    for chunk in jobs.chunks(batch) {
        let states: Vec<&str> = chunk.iter().map(|(_, _, state)| state.as_str()).collect();
        let choices = decider.choose_many(&states, &cfg.instructions, &options);
        for ((x, y, _), choice) in chunk.iter().zip(choices) {
            let choice = match choice {
                Ok(choice) => choice,
                Err(err) => {
                    tracing::debug!(x, y, error = %err, "laya_ca_cell_failed");
                    failures += 1;
                    continue;
                }
            };
            confidence_sum += choice.confidence;
            confidence_n += 1;

            if choice.confidence < cfg.min_confidence {
                held += 1;
                continue;
            }

            let label = draw_label(&choice, cfg.temperature, rng);
            let Some(swatch) = (match cfg.prompt {
                Prompt::Sunset => palette::index_of(&label),
                Prompt::Life => life::swatch_for_label(&label),
            }) else {
                tracing::debug!(x, y, label = %label, "laya_ca_unknown_label");
                failures += 1;
                continue;
            };

            if swatch == lattice.cell(*x, *y).swatch {
                unchanged += 1;
            } else {
                next.set(*x, *y, swatch);
                rewrites += 1;
            }
        }
    }

    let report = StepReport {
        cells: lattice.width() * lattice.height(),
        rewrites,
        held,
        unchanged,
        failures,
        unique_swatches: next.unique_swatches(),
        mean_confidence: if confidence_n == 0 {
            0.0
        } else {
            confidence_sum / confidence_n as f32
        },
        elapsed_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
    };

    tracing::info!(
        cells = report.cells,
        rewrites = report.rewrites,
        held = report.held,
        unchanged = report.unchanged,
        failures = report.failures,
        unique_swatches = report.unique_swatches,
        mean_confidence = report.mean_confidence,
        temperature = cfg.temperature,
        batch = cfg.batch.max(1),
        elapsed_ms = report.elapsed_ms,
        "laya_ca_step"
    );

    (next, report)
}

/// `temperature <= 1` keeps the argmax. Above that, weights become `p^(1/T)` and we draw.
fn draw_label(choice: &Choice, temperature: f32, rng: &mut u64) -> String {
    if temperature <= 1.0 || choice.probabilities.is_empty() {
        return choice.label.clone();
    }
    let weights: Vec<f32> = choice
        .probabilities
        .iter()
        .map(|(_, probability)| probability.max(1e-8).powf(1.0 / temperature))
        .collect();
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 {
        return choice.label.clone();
    }
    let mut cursor = (xorshift(rng) as f32 / u64::MAX as f32) * sum;
    for ((label, _), weight) in choice.probabilities.iter().zip(weights) {
        cursor -= weight;
        if cursor <= 0.0 {
            return label.clone();
        }
    }
    choice
        .probabilities
        .last()
        .map(|(label, _)| label.clone())
        .unwrap_or_else(|| choice.label.clone())
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state | 1;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decide::{Choice, DecideError, Decider};
    use crate::majority::MajorityDecider;
    use crate::palette;

    struct Fixed {
        label: &'static str,
        confidence: f32,
        fail: bool,
    }

    impl Decider for Fixed {
        fn choose(
            &self,
            _state: &str,
            _instructions: &str,
            _options: &[String],
        ) -> Result<Choice, DecideError> {
            if self.fail {
                return Err(DecideError::Model("offline".into()));
            }
            Ok(Choice {
                label: self.label.to_string(),
                confidence: self.confidence,
                probabilities: Vec::new(),
            })
        }
    }

    #[test]
    fn batched_steps_match_one_cell_at_a_time() {
        let lattice = Lattice::seeded(4, 4, 3).unwrap();
        let mut rng_one = 1u64;
        let mut rng_many = 1u64;
        let one = StepConfig {
            batch: 1,
            ..StepConfig::default()
        };
        let many = StepConfig {
            batch: 5,
            ..StepConfig::default()
        };
        let (left, left_report) = evolve(&lattice, &MajorityDecider, &one, &mut rng_one);
        let (right, right_report) = evolve(&lattice, &MajorityDecider, &many, &mut rng_many);
        assert_eq!(left, right);
        assert_eq!(left_report.rewrites, right_report.rewrites);
    }

    #[test]
    fn majority_of_neighbors_repaints_the_center() {
        let mut lattice = Lattice::filled(3, 3, 0).unwrap();
        let amber = palette::index_of("#ff0000").unwrap();
        for (x, y) in [
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
        ] {
            lattice.set(x, y, amber);
        }

        let (next, report) = evolve(&lattice, &MajorityDecider, &StepConfig::default(), &mut 1);

        assert_eq!(next.cell(1, 1).swatch, amber);
        assert_eq!(next.cell(0, 0).swatch, amber);
        assert_eq!(report.rewrites, 1);
        assert_eq!(report.failures, 0);
        assert!(report.mean_confidence > 0.9);
    }

    #[test]
    fn cell_may_keep_its_color_when_neighbors_differ() {
        let mut lattice = Lattice::filled(3, 3, 0).unwrap();
        let amber = palette::index_of("#ff0000").unwrap();
        lattice.set(1, 1, amber);
        let decider = Fixed {
            label: "#ff0000",
            confidence: 1.0,
            fail: false,
        };

        let (next, report) = evolve(&lattice, &decider, &StepConfig::default(), &mut 1);

        assert_eq!(next.cell(1, 1).swatch, amber);
        assert!(report.unchanged >= 1);
    }

    #[test]
    fn corner_only_counts_neighbors_inside_the_frame() {
        let mut lattice = Lattice::filled(3, 3, 0).unwrap();
        let violet = palette::index_of("#0000ff").unwrap();
        lattice.set(1, 0, violet);
        lattice.set(0, 1, violet);
        lattice.set(1, 1, violet);

        let (next, report) = evolve(&lattice, &MajorityDecider, &StepConfig::default(), &mut 1);

        assert_eq!(next.cell(0, 0).swatch, violet);
        assert_eq!(report.failures, 0);
    }

    #[test]
    fn model_failure_keeps_the_picture() {
        let mut lattice = Lattice::filled(3, 3, 0).unwrap();
        let amber = palette::index_of("#ff0000").unwrap();
        for y in 0..3 {
            for x in 0..3 {
                if (x + y) % 2 == 1 {
                    lattice.set(x, y, amber);
                }
            }
        }
        let decider = Fixed {
            label: "#ff0000",
            confidence: 1.0,
            fail: true,
        };

        let (next, report) = evolve(&lattice, &decider, &StepConfig::default(), &mut 1);

        assert_eq!(next, lattice);
        assert_eq!(report.failures, 9);
        assert_eq!(report.rewrites, 0);
        assert_eq!(report.mean_confidence, 0.0);
    }

    #[test]
    fn uniform_block_stays_put() {
        let lattice = Lattice::filled(3, 3, 0).unwrap();
        let (next, report) = evolve(&lattice, &MajorityDecider, &StepConfig::default(), &mut 1);
        assert_eq!(next, lattice);
        assert_eq!(report.unchanged, 9);
        assert_eq!(report.rewrites, 0);
    }

    #[test]
    fn low_confidence_holds_the_current_color() {
        let mut lattice = Lattice::filled(2, 2, 0).unwrap();
        lattice.set(1, 1, palette::index_of("#ff0000").unwrap());
        let decider = Fixed {
            label: "bright sunset orange",
            confidence: 0.01,
            fail: false,
        };
        let cfg = StepConfig {
            min_confidence: 0.5,
            ..StepConfig::default()
        };

        let (next, report) = evolve(&lattice, &decider, &cfg, &mut 1);

        assert_eq!(next, lattice);
        assert_eq!(report.held, 4);
        assert_eq!(report.rewrites, 0);
        assert_eq!(report.failures, 0);
    }

    #[test]
    fn unit_temperature_keeps_the_mode() {
        let choice = Choice {
            label: "#000000".into(),
            confidence: 0.99,
            probabilities: vec![("#000000".into(), 0.99), ("#ff0000".into(), 0.01)],
        };
        let mut rng = 1u64;
        for _ in 0..32 {
            assert_eq!(draw_label(&choice, 1.0, &mut rng), "#000000");
        }
    }

    #[test]
    fn flat_temperature_sometimes_draws_the_minority() {
        let choice = Choice {
            label: "#000000".into(),
            confidence: 0.99,
            probabilities: vec![("#000000".into(), 0.99), ("#ff0000".into(), 0.01)],
        };
        let mut rng = 1u64;
        let mut saw_red = false;
        for _ in 0..400 {
            if draw_label(&choice, 8.0, &mut rng) == "#ff0000" {
                saw_red = true;
                break;
            }
        }
        assert!(saw_red);
    }

    #[test]
    fn unknown_label_is_a_failure() {
        let mut lattice = Lattice::filled(2, 2, 0).unwrap();
        lattice.set(1, 1, palette::index_of("#ff0000").unwrap());
        let decider = Fixed {
            label: "not a color",
            confidence: 1.0,
            fail: false,
        };

        let (next, report) = evolve(&lattice, &decider, &StepConfig::default(), &mut 1);

        assert_eq!(next, lattice);
        assert_eq!(report.failures, 4);
    }
}
