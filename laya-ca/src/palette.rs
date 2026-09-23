use std::sync::OnceLock;

/// One legal cell color.
///
/// `label` is the single token Laya chooses. A `#rrggbb` option is several
/// tokens, and a 10×10 step pays that cost once per cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Swatch {
    pub hex: String,
    pub label: &'static str,
    pub rgba: [u8; 4],
}

/// Channel steps for a uniform quantization of the RGB cube.
///
/// Three levels give 27 colors. Four levels would be 64 options, past the
/// checkpoint's useful choice budget (`head_max_len` 256, and `choice:11+`
/// is already the wide bucket).
const LEVELS: [u8; 3] = [0x00, 0x80, 0xff];

/// One ModernBERT token each, in cube order. `aa` is the 27th.
const LABELS: [&str; 27] = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
    "t", "u", "v", "w", "x", "y", "z", "aa",
];

pub fn colors() -> &'static [Swatch] {
    static PALETTE: OnceLock<Vec<Swatch>> = OnceLock::new();
    PALETTE.get_or_init(quantize_spectrum)
}

pub fn label(index: usize) -> &'static str {
    colors()
        .get(index)
        .map(|swatch| swatch.label)
        .unwrap_or(LABELS[0])
}

pub fn index_of(label: &str) -> Option<usize> {
    let trimmed = label.trim().trim_matches('"');
    if let Some(index) = colors().iter().position(|swatch| swatch.label == trimmed) {
        return Some(index);
    }
    let normalized = normalize_hex(trimmed)?;
    colors().iter().position(|swatch| swatch.hex == normalized)
}

fn quantize_spectrum() -> Vec<Swatch> {
    let mut colors = Vec::with_capacity(LEVELS.len().pow(3));
    for red in LEVELS {
        for green in LEVELS {
            for blue in LEVELS {
                let index = colors.len();
                colors.push(Swatch {
                    hex: format!("#{red:02x}{green:02x}{blue:02x}"),
                    label: LABELS[index],
                    rgba: [red, green, blue, 255],
                });
            }
        }
    }
    colors
}

fn normalize_hex(label: &str) -> Option<String> {
    let trimmed = label.trim().trim_matches('"').to_ascii_lowercase();
    let hex = trimmed.strip_prefix('#').unwrap_or(&trimmed);
    if hex.len() != 6 || !hex.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return None;
    }
    Some(format!("#{hex}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectrum_is_the_27_color_cube() {
        let spectrum = colors();
        assert_eq!(spectrum.len(), 27);
        assert_eq!(spectrum[0].hex, "#000000");
        assert_eq!(spectrum.last().unwrap().hex, "#ffffff");
        assert_eq!(index_of("#FF0000"), index_of("ff0000"));
        assert_eq!(index_of(label(0)), Some(0));
        assert_eq!(index_of(label(26)), Some(26));
        assert_eq!(index_of(label(index_of("#ff0000").unwrap())), index_of("#ff0000"));
        assert!(spectrum.iter().any(|swatch| swatch.hex == "#ff0000"));
        assert!(spectrum.iter().any(|swatch| swatch.hex == "#00ff00"));
        assert!(spectrum.iter().any(|swatch| swatch.hex == "#0000ff"));
        assert!(index_of("not a color").is_none());
    }
}
