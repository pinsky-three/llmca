# LLMCA: Large Language Model Cellular Automata


<div align="center" width="100%">
<!--     <video src="https://github.com/user-attachments/assets/921f2b91-845f-4940-aa2f-fb1f191cdd8d" width="400" /> -->
    <table>
        <tr>
            <th><video src="https://github.com/user-attachments/assets/b3f6886f-1245-4303-a55c-da621c87c2d1" width="200" /></th>
            <th><video src="https://github.com/user-attachments/assets/4cd0d016-6783-4052-8db8-4c6e7429eeeb" width="200" /></th>
            <th><video src="https://github.com/user-attachments/assets/a0d7750d-8d25-438d-b4f7-c34503474a64" width="200" /></th>
            <th><video src="https://github.com/user-attachments/assets/7dc9a504-3cec-4a7f-8230-d6090ea15c5e" width="200" /></th>
            <th><video src="https://github.com/user-attachments/assets/4bc60a66-a53a-4e08-9dc2-db36fdb8f017" width="200" /></th>
            <th><video src="https://github.com/user-attachments/assets/483af1d5-7690-46da-9471-bde8f7158b13" width="200" /></th>
        </tr>
<!--<tr>
        <td><video src="https://github.com/user-attachments/assets/921f2b91-845f-4940-aa2f-fb1f191cdd8d" width="200" /></td>
        <td><video src="https://github.com/user-attachments/assets/921f2b91-845f-4940-aa2f-fb1f191cdd8d" width="200" /></td>
        <td><video src="https://github.com/user-attachments/assets/921f2b91-845f-4940-aa2f-fb1f191cdd8d" width="200" /></td>
      </tr>
      <tr>
        <td><video src="https://github.com/user-attachments/assets/921f2b91-845f-4940-aa2f-fb1f191cdd8d" width="200" /></td>
        <td><video src="https://github.com/user-attachments/assets/921f2b91-845f-4940-aa2f-fb1f191cdd8d" width="200" /></td>
        <td><video src="https://github.com/user-attachments/assets/921f2b91-845f-4940-aa2f-fb1f191cdd8d" width="200" /></td>
      </tr> -->
    </table>
</div>

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
