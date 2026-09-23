# laya-ca

A cellular automaton whose next color is a [Laya](https://github.com/aovestdipaperino/laya-rust) typed decision, not generated text.

Each pixel reads only its own previous color and its eight neighbors, then chooses its own next color from a 27-color quantization of the RGB cube (`#rrggbb`, three levels per channel: `00`, `80`, `ff`). Staying the same color is a legal update. `--temperature` (default `4`) flattens the 27-way distribution before sampling; `1` keeps the argmax, which locks. A decision below `--min-confidence` (default `0`) keeps the current color. A model error keeps the current color and increments `failures`.

Latency budget: one ModernBERT-large pass per cell. The default picture is 50×50, so one Laya generation is 2,500 passes. Sides above 50 are rejected.

## Weights

Use the fine-tuned checkpoint [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions). It is not committed (`models/` is gitignored). About 850 MB, Apache-2.0, no token:

```sh
mkdir -p models/laya-typed/encoder models/laya-typed/tokenizer
B=https://huggingface.co/convaiinnovations/laya-typed-decisions/resolve/main
for f in model.safetensors encoder/config.json tokenizer/tokenizer.json \
         tokenizer/tokenizer_config.json rl_agent_config.json; do
  curl -L --fail -o "models/laya-typed/$f" "$B/$f"
done
```

That checkpoint was trained on support, invoice, security, and agent-trace decisions. A sunset lattice is outside those workflows, so the picture is exploratory. The base checkpoint [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) loads the same way if you point `--model` at it.

## Run

```sh
# Apple GPU
cargo run -p laya-ca --release --features infer,metal -- --model models/laya-typed

# CPU
cargo run -p laya-ca --release --features infer -- --model models/laya-typed

# No checkpoint: each pixel becomes the majority neighbor phrase
cargo run -p laya-ca -- --fake
```

`LAYA_MODEL_DIR` is accepted in place of `--model`.

## Logs

`RUST_LOG=info` emits one JSON line per generation: cell count, rewrites, holds, unchanged cells, failures, unique colors, mean confidence, elapsed milliseconds. Prompts and answers are not logged at info. Cell failures are `debug`.

```sh
RUST_LOG=info cargo run -p laya-ca --release --features infer,metal -- --model models/laya-typed
```
