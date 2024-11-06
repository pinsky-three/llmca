# LLMCA: Large Language Model Cellular Automata


[![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](https://raw.githubusercontent.com/pinsky-three/llmca/main/saves/3f43596b26cc29274c67b271c7a840ef_1719215614.mp4)


LLMCA (Language Model Cellular Automata) is an experimental project that combines cellular automata with language models (LLM). This project allows simulating a cognitive space where each cell is a cognitive unit evolving based on rules defined by a language model.

## Features

- **Cellular Automata:** Simulation of cellular automata on a Von Neumann lattice.
- **Language Models:** Each cognitive unit interacts with its neighbors and decides its next state using a language model.
- **Distributed Computation:** The system is designed to distribute computation across different API instances (e.g., OpenAI, Ollama, etc.).
- **Visual Rendering:** The simulation is visually represented with cells in different colors, where the state of each cell is encoded in hexadecimal format.

## Requirements

- **Rust**: The project is primarily written in Rust, so you'll need Rust installed.
- **Cargo**: Required to manage the project dependencies.
- **OpenAI API Key**: You’ll need an API key from OpenAI or a compatible model API.
- **Macroquad**: Used for visualizing the simulation results.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/pinsky-three/llmca.git
    ```
2. Navigate to the project directory:
    ```bash
    cd llmca
    ```
3. Install the project dependencies with Cargo:
    ```bash
    cargo build
    ```

## Usage

1. Set up your environment variables:
    - Make sure you have a `.env` file with the following variables:
        ```bash
        OPENAI_API_URL=http://your_api_url
        OPENAI_MODEL_NAME=model_name
        OPENAI_API_KEY=your_api_key
        ```

2. Run the simulation:
    ```bash
    cargo run
    ```

3. The simulation will open a window displaying the evolution of the cellular automaton grid. The states of each cell are represented by colors based on their hexadecimal values.

## Simulation Example

Each cognitive unit follows rules based on language models. Below is an example of how a unit may decide its next state:

```json
{
  "rule": "Always respond with a hex string like: '#RRGGBB'",
  "state": ["#aaaaaa"],
  "neighbors": [
    { "n_0": ["#ff0000"] },
    { "n_1": ["#00ff00"] },
    { "n_2": ["#0000ff"] },
    { "n_3": ["#aaaaaa"] }
  ]
}
```

The language model responds with the next state in hexadecimal format.

## Contributions

Contributions are welcome! Feel free to fork this project and submit a pull request with your changes.

---

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](./LICENSE) file.
