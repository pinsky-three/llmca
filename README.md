
# LLMCA: Large Language Model Cellular Automata

[![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](saves/21f3eaccfb3841a35c2a60a597f06d50_1727826848.mp4)
[![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](saves/b0f16e50d0c469441c2e2e09ceab8bb4_1720554721.mp4)
[![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](saves/3f43596b26cc29274c67b271c7a840ef_1719038138.mp4)
[![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](saves/31f4debe9c9e2233dd6bd803614f5233_1728006302.mp4)
[![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](saves/4c42aeae66cac878b21b83a217a4928c_1719265698.mp4)
[![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](saves/21f3eaccfb3841a35c2a60a597f06d50_1727706668.mp4)
[![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](saves/b0d01a0d24c35780b62805c63e5fb573_1718658527.mp4)
[![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](saves/b0f16e50d0c469441c2e2e09ceab8bb4_1727470870.mp4)
[![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](saves/21f3eaccfb3841a35c2a60a597f06d50_1727641836.mp4)

<!-- <video controls src="saves/21f3eaccfb3841a35c2a60a597f06d50_1727826848.mp4" title="Title"></video> <video controls src="saves/b0f16e50d0c469441c2e2e09ceab8bb4_1720554721.mp4" title="Title"></video> <video controls src="saves/3f43596b26cc29274c67b271c7a840ef_1719038138.mp4" title="Title"></video> <video controls src="saves/31f4debe9c9e2233dd6bd803614f5233_1728006302.mp4" title="Title"></video> <video controls src="saves/4c42aeae66cac878b21b83a217a4928c_1719265698.mp4" title="Title"></video> <video controls src="saves/21f3eaccfb3841a35c2a60a597f06d50_1727706668.mp4" title="Title"></video> <video controls src="saves/b0d01a0d24c35780b62805c63e5fb573_1718658527.mp4" title="Title"></video> <video controls src="saves/b0f16e50d0c469441c2e2e09ceab8bb4_1727470870.mp4" title="Title"></video> <video controls src="saves/21f3eaccfb3841a35c2a60a597f06d50_1727641836.mp4" title="Title"></video>
 -->


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
