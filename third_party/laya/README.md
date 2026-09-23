# laya

Rust inference for [Laya](https://huggingface.co/convaiinnovations/laya), a non-autoregressive
typed-decision model of the kind TypeSafe calls a
[System One model](https://typesafe.ai/blog/introducing-system-one-models-and-jev). You give
it a **state** (text or JSON) and a set of **typed questions**; it returns typed answers with
calibrated probabilities in a single forward pass. It never
generates text, so there is nothing to parse and nothing to hallucinate.

Pure Rust on [candle](https://github.com/huggingface/candle) — no Python, no torch, no ONNX
export step. Runs on CPU, Metal, or CUDA.

## Getting the weights

The root checkpoint is Apache-2.0 and ungated, so no token is needed:

```sh
mkdir -p models/laya-base/encoder models/laya-base/tokenizer
B=https://huggingface.co/convaiinnovations/laya/resolve/main
for f in model.safetensors encoder/config.json tokenizer/tokenizer.json \
         tokenizer/tokenizer_config.json rl_agent_config.json; do
  curl -sL -o models/laya-base/$f "$B/$f"
done
```

That is ~847 MB. The repo also carries two sibling variants under
[`multilingual/`](https://huggingface.co/convaiinnovations/laya-multilingual) and
[`typed-decisions/`](https://huggingface.co/convaiinnovations/laya-typed-decisions); this crate
loads any of them, since the layout is identical, so point `--model` at the directory you
downloaded. `typed-decisions/` is the fine-tuned one, marked
`"fine_tuned": true` with temperatures near 1.0 rather than near 2.0, and it is the sensible
default for typed decisions.

## CLI

```sh
cargo build --release
./target/release/laya \
  --state-file examples/ticket.txt \
  --questions examples/questions.json
```

```json
{
  "model": "rl-agent",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.9641, "technical": 0.0174, "sales": 0.0185 },
      "confidence": 0.8365,
      "rl_agent": { "act_probability": 1.0 }
    },
    "urgency": {
      "type": "score",
      "score": 0.7031,
      "legend": { "0": "not urgent", "1": "low", "2": "normal", "3": "high", "4": "critical outage" },
      "probabilities": { "0": 0.6428, "1": 0.0763, "2": 0.23, "3": 0.0366, "4": 0.0142 },
      "confidence": 0.3786,
      "rl_agent": { "act_probability": 1.0 }
    },
    "needs_human": {
      "type": "noul",
      "noul": 0.7861,
      "rl_agent": { "act_probability": 1.0 }
    }
  },
  "usage": { "input_tokens": 233, "output_tokens": 0 }
}
```

## Web console

```sh
cargo run --release --features serve --bin laya-serve   # → http://127.0.0.1:8080
```

A local single-page console: the input on the left, the decision on the right. Each answer
shows the verdict, its probability, the confidence, and one line naming the runner-up — enough
to see a close call without reading a full distribution. **Table view** opens the rest: every
option's probability, the score legend, and `P(answer)`.

Two things it shows that the CLI cannot:

- **A prompt inspector**, collapsed under the answers. The exact token sequence the encoder
  saw, coloured by segment (instructions / options / state / structure) with each scored
  `[MASK]` marker ringed and numbered, plus how much of the `max_len` budget the state used —
  and a red *state truncated* warning on the collapsed header when it overflowed.
- **Presets** for support triage, content moderation, and agent-trace review.

`⌘/Ctrl + ↵` runs. Works in light and dark, and down to phone width. It binds to loopback only
and has no auth — it is a dev tool, not a deployment.

## Library

```rust
use laya::{Agent, Options, Question};
use serde_json::json;

let agent = Agent::from_dir("models/laya-base", Options::default())?;

let questions = vec![
    ("department".to_string(),
     Question::choice("Which team owns this?", ["billing", "technical", "sales"])),
    ("urgency".to_string(),
     Question::score("How urgent?", ["not urgent", "low", "normal", "high"])),
    ("needs_human".to_string(),
     Question::noul("A human must handle this.")),
];

let response = agent.system_one(&json!("Charged twice for order #4417."), &questions)?;
println!("{}", serde_json::to_string_pretty(&response)?);
# Ok::<(), anyhow::Error>(())
```

All questions for one state go through **a single batched forward pass**, so asking five
questions costs about as much as asking one.

## The three question types

| Type | `criteria` | Answer |
|---|---|---|
| `choice` | list of labels, or `{label: description}` | the argmax label plus a distribution over labels |
| `score` | list of level descriptions | the expectation over level indices, plus the distribution |
| `noul` | optional `{"false": ..., "true": ...}` | P(the statement holds) |

`choice` criteria keep the order you wrote them in, and that order is what the label
probabilities are reported against.

Every answer also carries `rl_agent.act_probability`: the act head's probability of answering
rather than escalating to a stronger model.

## Writing questions that work

The model is a text encoder scoring `[MASK]` markers by the meaning of the text around them.
Two habits follow from that, and between them they are worth more than the choice of
checkpoint.

**Give it prose, not a struct.** Passing the structured record you already have,
`{"command": "sudo rm -rf /", "cwd": "..."}`, hands the encoder a document with almost no
meaning in it. Name the situation the way a person would: *"I am working in the project folder
/Users/enzo/Code/plank. A coding assistant wants to run this command: sudo rm -rf /"*.

**Make options phrases, not tokens.** `["up", "stay", "down"]` and `["allow", "confirm",
"deny"]` are enum values, and `stay` barely means anything on its own. `["move the paddle up",
"keep the paddle still", "move the paddle down"]` means something, and on the pong demo that
single change takes the answer from 3/5 correct to 5/5.

**Ask about the world, not about your policy.** The model has not read your rules, so asking it
to pick the right bucket from a policy it cannot see is asking it to guess. Ask what is true
instead. A `noul` question, which returns the probability that a statement holds, is usually the
strongest shape available.

`examples/exfil_triage.rs` works all three through on a real problem: spotting a command that
sends a credential off the machine, which is the one thing a write-containing sandbox cannot
see, since reads and network calls are both allowed.

```sh
cargo run --release --example exfil_triage -- models/laya-typed
```

On a thirty-command held-out set it catches 10 of 11 at a fixed 0.5 threshold with one false
alarm, precision and recall both 0.91. Obfuscated variants score *higher* than plain ones,
so wrapping the payload in `base64` or `$(cat ...)` does not get past it. Read-only commands
like `cat ~/.ssh/id_rsa` sit below the line, which is correct: the risk is a read plus a send.

One field not to reach for: `rl_agent.act_probability` reads `1.0000` for every input on the
published checkpoints, so use the entropy-derived `confidence` instead.

## How it works

```mermaid
flowchart LR
  S["state<br/>(text or JSON)"] --> B
  Q["typed question"] --> B
  B["[CLS] type question: instructions [SEP]<br/>[MASK] opt0 [MASK] opt1 … [SEP]<br/>state [SEP]"] --> E
  E["ModernBERT-large<br/>28 layers, 1024 hidden"] --> T["+ question-type embedding"]
  T --> H["decision head<br/>2 pre-norm transformer layers"]
  H --> G["gather the [MASK] marker<br/>at each option"]
  G --> SC["scorer → one logit per option"]
  SC --> TS["÷ temperature<br/>(per type and option count)"] --> P["calibrated distribution"]
  H --> AP["pooled [CLS]"] --> AH["act head"] --> AC["P(answer) vs P(escalate)"]
  SC -.->|"top-1, margin, entropy, k"| AH
```

Each option gets a `[MASK]` marker in the prompt; the head scores those marker positions and
softmaxes over them. That is why the whole question set resolves in one pass and why the model
cannot emit anything outside your option list.

## Configuration

`rl_agent_config.json` sets the budgets: `max_len` 512 and `head_max_len` 192 on the root
checkpoint. Options share the head budget, so a question with many long options will fail with
an explicit `head_max_len` error rather than silently dropping options. For more than ~20
options, raise both budgets (the encoder supports up to 8192 positions) or split the question
into a coarse-to-fine pair.

## Backends

```sh
cargo build --release                       # CPU, library + laya CLI
cargo build --release --features serve      # also the laya-serve web console
cargo build --release --features metal     # Apple GPU
cargo build --release --features cuda      # NVIDIA
```

f32 by default, f16 on an accelerator with `--dtype f16`. The weights are f16 on disk, so f32
doubles them to 1.69 GB resident where f16 keeps them at 0.84 GB. Upstream candle's ModernBERT
builds both its attention masks as f32 unconditionally, which makes an f16 backbone fail inside
the first attention block; `src/modernbert.rs` is that file vendored with the masks following
the model's dtype instead. Answers agree with f32 to about 1e-3, so the one thing f16 costs is
bit-exact agreement between a batched answer and the same question asked on its own. CPU is f32
only: candle has no f16 CPU kernels.

## Calibration

The checkpoint ships over-confident. The bundled temperatures move mean ECE from 0.466 to
0.081 on the authors' data, but those were fitted on *their* distribution. Refit one
temperature per (question type, option count) bucket on your own data before trusting the
probabilities as probabilities; `temperature_by_options` in `rl_agent_config.json` is the knob,
keyed `choice:3-5`, `noul:2`, and so on.

Also worth knowing before you build on it: the base checkpoint is near chance on the authors'
own typed-decisions benchmark (0.362 against a 0.461 majority-class baseline). The strong
numbers belong to the fine-tuned `typed-decisions` variant. Treat this as a fast base to
specialise, not a zero-shot decision engine.

## Tests

```sh
cargo test --release
```

The rendering tests run anywhere. The inference tests skip themselves unless a checkpoint is
present at `models/laya-base` (override with `LAYA_MODEL_DIR`).

## License

Apache-2.0, matching the upstream model. The console's header image is supplied by the
repository owner and is not covered by that license.

