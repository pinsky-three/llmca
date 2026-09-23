use crate::palette;

/// Why a lattice could not be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LatticeError {
    #[error("lattice size must be between 1 and {max}, got {width}x{height}")]
    Size {
        width: usize,
        height: usize,
        max: usize,
    },
    #[error("swatch index {index} is outside the palette")]
    Swatch { index: usize },
}

/// One pixel. The index is into [`palette::colors`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub swatch: usize,
}

/// Row-major picture. Evolution reads it and writes a new one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lattice {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
}

const MAX_SIDE: usize = 50;

const NEIGHBORS: &[(isize, isize, &str)] = &[
    (0, -1, "north"),
    (1, -1, "northeast"),
    (1, 0, "east"),
    (1, 1, "southeast"),
    (0, 1, "south"),
    (-1, 1, "southwest"),
    (-1, 0, "west"),
    (-1, -1, "northwest"),
];

impl Lattice {
    pub fn filled(width: usize, height: usize, swatch: usize) -> Result<Self, LatticeError> {
        check_size(width, height)?;
        if swatch >= palette::colors().len() {
            return Err(LatticeError::Swatch { index: swatch });
        }
        Ok(Self {
            width,
            height,
            cells: vec![Cell { swatch }; width * height],
        })
    }

    /// Reproducible noise. The same seed always paints the same picture.
    pub fn seeded(width: usize, height: usize, seed: u64) -> Result<Self, LatticeError> {
        check_size(width, height)?;
        let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let cells = (0..width * height)
            .map(|_| {
                rng = xorshift(rng);
                Cell {
                    swatch: (rng as usize) % palette::colors().len(),
                }
            })
            .collect();
        Ok(Self {
            width,
            height,
            cells,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn cell(&self, x: usize, y: usize) -> Cell {
        self.cells[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, swatch: usize) {
        self.cells[y * self.width + x].swatch = swatch;
    }

    pub fn unique_swatches(&self) -> usize {
        let mut seen = vec![false; palette::colors().len()];
        let mut count = 0;
        for cell in &self.cells {
            if let Some(slot) = seen.get_mut(cell.swatch) {
                if !*slot {
                    *slot = true;
                    count += 1;
                }
            }
        }
        count
    }

    /// Prose for one pixel. Laya scores text meaning, so a JSON record of
    /// neighbor ids is a worse state than a sentence a person would say.
    ///
    /// The sentences stay short on purpose. A 10×10 step is a hundred full
    /// encoder passes, and the pass time grows with the token count.
    pub fn describe(&self, x: usize, y: usize) -> String {
        let mut lines = Vec::with_capacity(2 + NEIGHBORS.len());
        lines.push(format!(
            "sunset column {} row {} was {}",
            x + 1,
            y + 1,
            self.phrase_at(x, y)
        ));
        for (dx, dy, name) in NEIGHBORS {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            match self.phrase_at_signed(nx, ny) {
                Some(phrase) => lines.push(format!("{name} {phrase}")),
                None => lines.push(format!("{name} edge")),
            }
        }
        lines.join(" ")
    }

    fn phrase_at(&self, x: usize, y: usize) -> &'static str {
        palette::label(self.cell(x, y).swatch)
    }

    fn phrase_at_signed(&self, x: isize, y: isize) -> Option<&'static str> {
        if x < 0 || y < 0 {
            return None;
        }
        let (x, y) = (x as usize, y as usize);
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.phrase_at(x, y))
    }
}

fn check_size(width: usize, height: usize) -> Result<(), LatticeError> {
    if (1..=MAX_SIDE).contains(&width) && (1..=MAX_SIDE).contains(&height) {
        Ok(())
    } else {
        Err(LatticeError::Size {
            width,
            height,
            max: MAX_SIDE,
        })
    }
}

fn xorshift(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_lattice_is_stable() {
        let a = Lattice::seeded(4, 4, 7).unwrap();
        let b = Lattice::seeded(4, 4, 7).unwrap();
        assert_eq!(a, b);
        assert!(Lattice::seeded(0, 4, 1).is_err());
        assert!(Lattice::seeded(50, 50, 1).is_ok());
        assert!(Lattice::seeded(4, 51, 1).is_err());
    }

    #[test]
    fn corner_prose_names_the_frame_edge() {
        let lattice = Lattice::filled(3, 3, 0).unwrap();
        let text = lattice.describe(0, 0);
        assert!(text.contains("sunset column 1 row 1 was a"));
        assert!(text.contains("north edge"));
        assert!(text.contains("west edge"));
        assert!(text.contains(" east a"));
        assert!(!text.contains(" east edge"));
    }
}
