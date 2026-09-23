//! Conway's Game of Life on the same lattice the color automaton uses.
//!
//! A missing neighbor is dead, matching the frame edge in the prose. Live is
//! `#ffffff` and every other swatch is dead, so a Life probe only needs two options.

use crate::lattice::Lattice;
use crate::palette;

pub const DEAD: &str = "#000000";
pub const LIVE: &str = "#ffffff";

/// Routing table in the question. The state is only the situation key, so the
/// option words `on` and `off` are not already sitting in the cell's sentence.
pub const LIFE_INSTRUCTIONS: &str = "\
Which next mark should this cell have? \
occupied with 2 neighbors: on. \
occupied with 3 neighbors: on. \
empty with exactly 3 neighbors: on. \
Every other situation: off.";

const OFFSETS: &[(isize, isize, &str)] = &[
    (0, -1, "north"),
    (1, -1, "northeast"),
    (1, 0, "east"),
    (1, 1, "southeast"),
    (0, 1, "south"),
    (-1, 1, "southwest"),
    (-1, 0, "west"),
    (-1, -1, "northwest"),
];

/// How close one generation came to the exact Life update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifeScore {
    pub cells: usize,
    pub matches: usize,
    /// Cells whose Life update differs from the current color.
    pub should_change: usize,
    pub changed_correct: usize,
    pub should_stay: usize,
    pub stayed_correct: usize,
    pub exact: bool,
}

pub fn options() -> Vec<String> {
    vec!["off".to_string(), "on".to_string()]
}

/// Map a Life decision back onto the cube. Hex labels still count.
pub fn swatch_for_label(label: &str) -> Option<usize> {
    match label.trim().to_ascii_lowercase().as_str() {
        "on" | "live" | "#ffffff" => Some(live_swatch()),
        "off" | "dead" | "#000000" => Some(dead_swatch()),
        other => palette::index_of(other),
    }
}

pub fn describe(lattice: &Lattice, x: usize, y: usize) -> String {
    let word = if is_live(lattice, x, y) {
        "occupied"
    } else {
        "empty"
    };
    let neighbors = live_neighbors(lattice, x, y);
    let count = if word == "empty" && neighbors == 3 {
        "exactly 3".to_string()
    } else {
        neighbors.to_string()
    };
    format!("The situation is {word} with {count} neighbors.")
}

/// One Life generation. The caller owns the synchronous update.
pub fn step(lattice: &Lattice) -> Lattice {
    let dead = dead_swatch();
    let live = live_swatch();
    let mut next = lattice.clone();
    for y in 0..lattice.height() {
        for x in 0..lattice.width() {
            let neighbors = live_neighbors(lattice, x, y);
            let alive = lattice.cell(x, y).swatch == live;
            let stays = if alive {
                neighbors == 2 || neighbors == 3
            } else {
                neighbors == 3
            };
            next.set(x, y, if stays { live } else { dead });
        }
    }
    next
}

/// Score `predicted` against the Life update of `before`.
///
/// Overall match rate is a weak number on a mostly dead board: a model that
/// never rewrites still matches every cell that Life leaves alone.
pub fn score(before: &Lattice, predicted: &Lattice) -> LifeScore {
    let truth = step(before);
    let live = live_swatch();
    let mut matches = 0;
    let mut should_change = 0;
    let mut changed_correct = 0;
    let mut should_stay = 0;
    let mut stayed_correct = 0;
    let cells = before.width() * before.height();
    for y in 0..before.height() {
        for x in 0..before.width() {
            let was_live = before.cell(x, y).swatch == live;
            let truth_live = truth.cell(x, y).swatch == live;
            let pred_live = predicted.cell(x, y).swatch == live;
            if pred_live == truth_live {
                matches += 1;
            }
            if was_live == truth_live {
                should_stay += 1;
                if pred_live == truth_live {
                    stayed_correct += 1;
                }
            } else {
                should_change += 1;
                if pred_live == truth_live {
                    changed_correct += 1;
                }
            }
        }
    }
    LifeScore {
        cells,
        matches,
        should_change,
        changed_correct,
        should_stay,
        stayed_correct,
        exact: matches == cells,
    }
}

pub fn with_center(occupied: bool, neighbors: u8) -> Lattice {
    let mut lattice = Lattice::filled(3, 3, dead_swatch()).expect("3x3");
    let around = [
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 1),
    ];
    for (x, y) in around.into_iter().take(usize::from(neighbors.min(8))) {
        lattice.set(x, y, live_swatch());
    }
    if occupied {
        lattice.set(1, 1, live_swatch());
    }
    lattice
}

pub fn blinker() -> Lattice {
    let mut lattice = Lattice::filled(5, 5, dead_swatch()).expect("5x5");
    for y in 1..4 {
        lattice.set(2, y, live_swatch());
    }
    lattice
}

pub fn block() -> Lattice {
    let mut lattice = Lattice::filled(4, 4, dead_swatch()).expect("4x4");
    lattice.set(1, 1, live_swatch());
    lattice.set(2, 1, live_swatch());
    lattice.set(1, 2, live_swatch());
    lattice.set(2, 2, live_swatch());
    lattice
}

pub struct Probe {
    pub name: &'static str,
    pub generations: u32,
    pub lattice: Lattice,
}

pub fn probes() -> Vec<Probe> {
    vec![
        Probe {
            name: "blinker",
            generations: 4,
            lattice: blinker(),
        },
        Probe {
            name: "block",
            generations: 2,
            lattice: block(),
        },
        Probe {
            name: "glider",
            generations: 4,
            lattice: glider(),
        },
    ]
}
/// Random binary field. About three cells in eight start live.
pub fn soup(
    width: usize,
    height: usize,
    seed: u64,
) -> Result<Lattice, crate::lattice::LatticeError> {
    let mut lattice = Lattice::filled(width, height, dead_swatch())?;
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for y in 0..height {
        for x in 0..width {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            if rng % 8 < 3 {
                lattice.set(x, y, live_swatch());
            }
        }
    }
    Ok(lattice)
}

pub fn centered_blinker(
    width: usize,
    height: usize,
) -> Result<Lattice, crate::lattice::LatticeError> {
    let mut lattice = Lattice::filled(width, height, dead_swatch())?;
    if width < 3 || height < 3 {
        return Ok(lattice);
    }
    let x = width / 2;
    let y = height / 2;
    lattice.set(x, y - 1, live_swatch());
    lattice.set(x, y, live_swatch());
    lattice.set(x, y + 1, live_swatch());
    Ok(lattice)
}

pub fn glider() -> Lattice {
    let mut lattice = Lattice::filled(10, 10, dead_swatch()).expect("10x10");
    for (x, y) in [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)] {
        lattice.set(x + 2, y + 2, live_swatch());
    }
    lattice
}

fn dead_swatch() -> usize {
    palette::index_of(DEAD).expect("black is in the cube")
}

fn live_swatch() -> usize {
    palette::index_of(LIVE).expect("white is in the cube")
}

fn is_live(lattice: &Lattice, x: usize, y: usize) -> bool {
    lattice.cell(x, y).swatch == live_swatch()
}

fn neighbor_live(lattice: &Lattice, x: isize, y: isize) -> Option<bool> {
    if x < 0 || y < 0 {
        return None;
    }
    let (x, y) = (x as usize, y as usize);
    if x >= lattice.width() || y >= lattice.height() {
        return None;
    }
    Some(is_live(lattice, x, y))
}

fn live_neighbors(lattice: &Lattice, x: usize, y: usize) -> u8 {
    OFFSETS
        .iter()
        .filter(|(dx, dy, _)| {
            neighbor_live(lattice, x as isize + dx, y as isize + dy).unwrap_or(false)
        })
        .count() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_with_three_neighbors_is_born() {
        let start = with_center(false, 3);
        assert_eq!(
            describe(&start, 1, 1),
            "The situation is empty with exactly 3 neighbors."
        );
        assert_eq!(step(&start).cell(1, 1).swatch, live_swatch());
        assert_eq!(swatch_for_label("on"), Some(live_swatch()));
        assert_eq!(swatch_for_label("off"), Some(dead_swatch()));
    }

    #[test]
    fn blinker_has_period_two() {
        let start = blinker();
        let mid = step(&start);
        assert_ne!(mid, start);
        assert_eq!(mid.cell(1, 2).swatch, live_swatch());
        assert_eq!(mid.cell(2, 2).swatch, live_swatch());
        assert_eq!(mid.cell(3, 2).swatch, live_swatch());
        assert_eq!(mid.cell(2, 1).swatch, dead_swatch());
        assert_eq!(step(&mid), start);
    }

    #[test]
    fn block_is_a_still_life() {
        let start = block();
        assert_eq!(step(&start), start);
        let score = score(&start, &start);
        assert!(score.exact);
        assert_eq!(score.should_change, 0);
        assert_eq!(score.stayed_correct, score.should_stay);
    }

    #[test]
    fn glider_translates_southeast_after_four_generations() {
        let start = glider();
        let mut lattice = start.clone();
        for _ in 0..4 {
            lattice = step(&lattice);
        }
        for y in 0..start.height() {
            for x in 0..start.width() {
                let expected = if x > 0 && y > 0 {
                    start.cell(x - 1, y - 1).swatch
                } else {
                    dead_swatch()
                };
                assert_eq!(lattice.cell(x, y).swatch, expected, "cell {x},{y}");
            }
        }
    }

    #[test]
    fn staying_put_misses_the_blinker_births() {
        let start = blinker();
        let score = score(&start, &start);
        assert!(!score.exact);
        assert_eq!(score.changed_correct, 0);
        assert_eq!(score.should_change, 4);
        assert_eq!(score.stayed_correct, score.should_stay);
    }

    #[test]
    fn soup_is_binary_and_stable() {
        let a = soup(8, 6, 3).unwrap();
        let b = soup(8, 6, 3).unwrap();
        assert_eq!(a, b);
        assert!(soup(0, 4, 1).is_err());
        let mut live = 0;
        for y in 0..a.height() {
            for x in 0..a.width() {
                let swatch = a.cell(x, y).swatch;
                assert!(swatch == dead_swatch() || swatch == live_swatch());
                if swatch == live_swatch() {
                    live += 1;
                }
            }
        }
        assert!(live > 0 && live < 48);
    }

    #[test]
    fn centered_blinker_is_away_from_the_edge() {
        let start = centered_blinker(8, 8).unwrap();
        let next = step(&start);
        assert_ne!(next, start);
        assert_eq!(next.cell(3, 4).swatch, live_swatch());
        assert_eq!(next.cell(4, 4).swatch, live_swatch());
        assert_eq!(next.cell(5, 4).swatch, live_swatch());
        assert_eq!(step(&next), start);
    }

    #[test]
    fn center_of_the_blinker_states_its_count() {
        let text = describe(&blinker(), 2, 2);
        assert!(text.contains("The situation is occupied with 2 neighbors."));
    }
}
