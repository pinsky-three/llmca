/// Closed set of questions a pixel may ask about its own next color.
///
/// The color update is a pure function of the neighborhood and this question.
/// A fixed question therefore has fixed points. Changing the question is what
/// lets a settled pixel move again.
pub const RULES: &[&str] = &[
    "What color is this pixel in the next moment?",
    "Which touching color is the brightest?",
    "Which touching color is the darkest?",
    "Which touching color is the warmest?",
    "Which touching color is the coolest?",
];

pub const NEXT_RULE: &str = "Which question should this pixel ask about its own next color?";

pub fn rule(index: usize) -> &'static str {
    RULES.get(index).copied().unwrap_or(RULES[0])
}

pub fn labels() -> Vec<String> {
    RULES.iter().map(|rule| (*rule).to_string()).collect()
}

pub fn index_of(label: &str) -> Option<usize> {
    let label = label.trim();
    RULES.iter().position(|rule| *rule == label)
}

pub fn advance(index: usize) -> usize {
    (index + 1) % RULES.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_are_closed_and_not_substrings() {
        assert!(RULES.len() >= 2);
        for (i, left) in RULES.iter().enumerate() {
            assert_eq!(index_of(left), Some(i));
            for (j, right) in RULES.iter().enumerate() {
                if i != j {
                    assert!(!left.contains(right));
                }
            }
        }
        assert_eq!(advance(RULES.len() - 1), 0);
    }
}
