# LLMCA: Large Language Model Cellular Automata


<div align="center" width="100%">
    <table>
        <tr>
            <th><video src="https://github.com/user-attachments/assets/b3f6886f-1245-4303-a55c-da621c87c2d1" width="200" /></th>
            <th><video src="https://github.com/user-attachments/assets/4cd0d016-6783-4052-8db8-4c6e7429eeeb" width="200" /></th>
            <th><video src="https://github.com/user-attachments/assets/a0d7750d-8d25-438d-b4f7-c34503474a64" width="200" /></th>
            <th><video src="https://github.com/user-attachments/assets/7dc9a504-3cec-4a7f-8230-d6090ea15c5e" width="200" /></th>
            <th><video src="https://github.com/user-attachments/assets/4bc60a66-a53a-4e08-9dc2-db36fdb8f017" width="200" /></th>
            <th><video src="https://github.com/user-attachments/assets/483af1d5-7690-46da-9471-bde8f7158b13" width="200" /></th>
        </tr>
    </table>
</div>

LLMCA (Language Model Cellular Automata) is an experimental project that combines cellular automata with large language models (LLMs).  It simulates a cognitive space where each cell evolves based on rules defined and interpreted by an LLM, considering the state of its neighbors.

## Features

- **Cognitive Units:**  Each cell acts as a cognitive unit with memory, storing its past states.
- **LLM-Driven Evolution:**  Cells determine their next state by querying an LLM, providing their history and their neighbors' current states. The LLM responds with a new state and optionally, an updated rule.
- **Von Neumann Neighborhood:**  Cells interact with their immediate neighbors in a 2D grid using the Von Neumann neighborhood (north, south, east, west, and diagonals).
- **Distributed Computation:** Supports distributing computations across multiple LLM API instances for improved performance.
- **Visualization:**  Renders the simulation in real-time using Macroquad, representing cell states with colors derived from hexadecimal strings returned by the LLM.
- **Persistence:** Saves the simulation state to disk, allowing resumption from previous steps.


## Requirements

- **Rust & Cargo:**  Ensure you have Rust and Cargo installed.
- **LLM API Access:**  Requires access to an LLM API compatible with the OpenAI API format (e.g., OpenAI, Ollama). Set up necessary environment variables (see Usage).
- **Macroquad:** For visualization.


## Installation

1. **Clone:** `git clone https://github.com/pinsky-three/llmca.git`
2. **Build:** `cd llmca && cargo build`

## Usage

1. **Environment Variables:** Create a `.env` file in the project root with the following:

   ```bash
   OPENAI_API_URL="http://your_api_url:port/v1" # Comma-separated for multiple APIs
   OPENAI_MODEL_NAME="your_model_name" # Comma-separated for multiple models
   OPENAI_API_KEY="your_api_key" # Comma-separated for multiple keys
   ```
   If using multiple APIs, ensure the number of URLs, model names, and API keys match.

2. **Run:** `cargo run -p minimal-ui`

## Simulation Example

The LLM receives a JSON input representing a cell's memory (previous states) and its neighbors' current states.  It's instructed to return a JSON object containing the next state and optionally, a new rule.

**Example LLM Input (Simplified):**

```json
[
  "self memory",
  {"rule": "be red if neighbors are green", "state": "#ff0000"},
  "neighbors",
  {"rule": "...", "state": "#00ff00"},
  {"rule": "...", "state": "#00ff00"} 
]
```

**Example LLM Output:**

```json
{"rule": "be red if neighbors are green", "state": "#ff0000"}
```

The visualization then interprets the `state` (e.g., `#ff0000`) as a color.


## Contributions

Contributions are welcome!  Fork the project and submit pull requests.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](./LICENSE) file.
