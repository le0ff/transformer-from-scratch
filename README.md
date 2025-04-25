# Transformer from Scratch (NumPy Implementation)

A NumPy-only implementation of the Transformer architecture ("Attention Is All You Need"), built from the ground up to understand the Transformer architecture and its various components.

## Table of Contents

* [Project Goal](#project-goal)
* [Core Idea & Approach](#core-idea--approach)
* [Implemented Components](#implemented-components)
* [Project Structure](#project-structure)
* [Setup & Installation](#setup--installation)
* [Usage](#usage)
    * [Running Tests](#running-tests)
    * [Forward Pass](#forward-pass)
* [Current Status & Limitations](#current-status--limitations)
* [Future Work](#future-work)
* [References](#references)
* [Project Members](#project-members)

## Project Goal

This project is the final assignment for the Advanced NLP course. The primary goal was to gain a deep understanding of the Transformer architecture by implementing it entirely from scratch using only the NumPy library, without relying on high-level deep learning frameworks like PyTorch or TensorFlow.

The initial objective was to build a character-level Transformer capable of learning simple sequence-to-sequence tasks, such as the identity function (`apple -> apple`) or the reverse function (`apple -> elppa`), as a proof-of-concept for the model's learning capabilities.

## Core Idea & Approach

We adopted a modular design approach:

1.  **NumPy Exclusivity (Initial Focus):** All mathematical operations and tensor manipulations are handled purely by NumPy. This was a deliberate choice to focus on the fundamental algorithms without framework abstractions.
2.  **Modular Components:** The Transformer architecture is broken down into its fundamental building blocks (e.g., Multi-Head Attention, Feed-Forward Blocks, Input Embedding, Positional Encoding, ...).
3.  **Base Layer:** A `BaseLayer` class (`src/layers/base.py`) was created, serving as an abstract parent class for all components. Each specific layer (e.g., `Linear`, `Softmax`, `MultiHeadAttentionBlock`) inherits from this base, ensuring a consistent interface, so all layers have or override methods such as `forward`, `get_parameters`, `set_parameters`, toggles for `train`/`eval` and a planned `backward`.
4.  **Character-Level Focus:** The intended application was simplified to character-level tasks to manage complexity.
5.  **Consideration of JAX:** While sticking to NumPy for the initial implementation, we acknowledged its potential performance limitations, especially for larger models or datasets. We considered using [JAX](https://github.com/google/jax) as a potential alternative. JAX offers a NumPy-like API but includes benefits like Just-In-Time (JIT) compilation for significant speedups on accelerators (GPUs/TPUs) and powerful automatic differentiation capabilities (`jax.grad`). However, to maximize focus on the manual implementation details first, we deferred a potential JAX integration, and instead focused on the modular NumPy-only implementation.

## Implemented Components

The following core components of the Transformer architecture have been implemented from scratch using NumPy (primarily focusing on the forward pass):

- **Activation Functions:**  
  - ReLU, Softmax

- **Embedding Layers:**  
  - Input Embedding, Positional Encoding

- **Core Layers:**  
  - Linear (fully connected), Dropout, Layer Normalization, Residual Connection

- **Attention Mechanism:**  
  - Multi-Head Attention Block

- **Feed-Forward Network:**  
  - Position-wise Feed-Forward Block

- **Model Blocks:**  
  - Encoder Block, Decoder Block (stacked to form Encoder and Decoder modules)

- **Tokenizer:**  
  - Simple character-level tokenizer for sequence tasks

- **Main Model:**  
  - Assembled Transformer class combining all components

All components are implemented as modular, extensible classes, following a unified interface via a custom abstract base class `BaseLayer`.

## Project Structure

The project follows a standard structure:

```
transformer-from-scratch/
│
├── experiments/              # Scripts for running experiments, demos, or exploration
│   ├── explore_forward.ipynb # Jupyter notebook for exploring the forward pass
│   └── forward_pass.py       
│
├── src/                      # Source code for the Transformer implementation
│   ├── layers/               # Core building blocks (layers) of the Transformer
│   │   ├── activations/      # Activation functions (e.g., ReLU, Softmax)
│   │   │   ├── __init__.py
│   │   │   ├── relu.py
│   │   │   └── softmax.py
│   │   ├── embeddings/       # Embedding layers (Input, Positional)
│   │   │   ├── __init__.py
│   │   │   ├── input_embedding.py
│   │   │   └── positional_encoding.py
│   │   ├── utils/            # Utility functions, e.g., for creating masks
│   │   │   ├── __init__.py
│   │   │   └── mask.py
│   │   ├── __init__.py       
│   │   ├── base.py           # Abstract Base Layer class for all components
│   │   ├── decoderblock.py   # Single block for the Decoder stack
│   │   ├── dropout.py        # Dropout layer
│   │   ├── encoderblock.py   # Single block for the Encoder stack
│   │   ├── feedforward.py    # Position-wise Feed-Forward Network layer
│   │   ├── linear.py         # Linear (Dense) layer
│   │   ├── multiheadattentionblock.py # Multi-Head Attention mechanism
│   │   ├── normalization.py  # Layer Normalization
│   │   ├── projection.py     # Final Linear projection layer (after Decoder)
│   │   └── residual.py       # Residual connection logic
│   │
│   ├── __init__.py           
│   ├── decoder.py            # The complete Decoder stack (multiple DecoderBlocks)
│   ├── encoder.py            # The complete Encoder stack (multiple EncoderBlocks)
│   ├── tokenizer.py          # Simple character-level tokenizer
│   └── transformer.py        # Main Transformer class assembling Encoder and Decoder
│
├── tests/                    # Unit tests for individual components
│   ├── layers/               # Tests mirroring the src/layers structure
│   │   ├── activations/      # Tests for activation functions
│   │   ├── embeddings/       # Tests for embedding layers
│   │   ├── __init__.py
│   │   ├── test_decoder_block.py
│   │   ├── test_dropout.py
│   │   ├── test_encoder_block.py
│   │   ├── test_feedforward.py
│   │   ├── test_linear.py
│   │   ├── test_multiheadattentionblock.py
│   │   ├── test_normalization.py
│   │   ├── test_projection.py
│   │   └── test_residual.py
│   │
│   ├── __init__.py           
│   ├── test_decoder.py
│   ├── test_encoder.py
│   ├── test_tokenizer.py
│   └── test_transformer.py   
│
├── .gitignore                
├── .python-version           # Specifies Python version (can be used via `uv python install`)
├── pyproject.toml            # Project metadata, build system, dependencies (`uv sync`)
├── README.md                 # Project documentation (you are reading this)
└── uv.lock                   # Lock file for dependencies (if using uv)
```

## Setup & Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/le0ff/transformer-from-scratch.git
    cd transformer-from-scratch
    ```

2. **Install [UV](https://docs.astral.sh/uv/getting-started/installation/)**
3. **Install dependencies:**
    ```bash
    uv sync
    ```

## Usage

### Running Tests

Unit tests are provided to verify the correctness of individual components.

- To run all tests:
```bash
uv run -m pytest
```

- To run specific test, for example `tests/layers/test_feedforward.py`:
```bash
uv run -m pytest tests/layers/test_feedforward.py
```

### Forward Pass

While the Transformer has not yet been trained, we have fully implemented and verified the forward pass. In our examples, we use a string as input and expect its inverse as output, processed as a batch using teacher forcing. With randomly initialized weights, the output is not semantically meaningful yet, but the pipeline and shapes are correct.

You can explore the forward pass in two ways:

- **Interactive Exploration:**  
  Open `experiments/explore_forward.ipynb` and execute the cells to see how the forward pass works, inspect tokens, batches and masks, and observe tensor shapes.

- **Script Execution:**  
  Run the following command to execute the forward pass via script (note: output is minimal and intended for verification):
  ```bash
  uv run -m experiments.forward_pass
  ```


## Current Status & Limitations

**Project Deadline:** April 25th, 5:00 PM CET

As of the current state, the project has achieved the following:
- **Modular Implementation:** Core Transformer layers are implemented as distinct modules inheriting from an abstract base class `BaseLayer`.
- **NumPy Core:** All computations and formulas are implemented  solely using NumPy.
- **Random Initialization & Reproducibility:** All components that require randomness use independent NumPy random number generators. If a seed is provided, initialization and all stochastic operations are fully reproducible: subcomponents deterministically derive their own seeds from the main seed, ensuring independent but repeatable results. If no seed is given, initialization is random. This approach avoids global side effects from `np.random.seed()` and ensures reproducibility without interfering with other code.
- **Functional Forward Pass:** The primary focus has been on ensuring that the forward pass of the implemented layers and the overall Transformer architecture works correctly — that is, producing outputs of the expected shapes and types. At the same time, the goal has been to gain a deep understanding of the Transformer architecture, including what each component does under the hood and how they work together as a complete pipeline.
- **Unit Tests:** Tests for initialization and functionality of several components are available.

### Limitations:
- **Backward Pass:** The `backward` pass for gradient computation is incomplete/only sketched for most layers at the moment. It has not been fully implemented or tested due to time constraints.
- **Training:** There is no training loop, optimizer, or loss function implemented. The model cannot be trained on any task yet.
- **Performance:** The pure NumPy implementation is not optimized for speed and can be slow compared to framework-based implementations or JIT-compiled code.
- **Task Execution:** The model has not been demonstrated on the target character-level tasks (identity, reverse).

## Future Work

Given more time, the following steps would be priorities:
1. **Complete NumPy Backward Pass:** Fully implement and rigorously test the `backward` method for all layers in NumPy, ensuring correct gradient calculations.
2. **Implement NumPy Optimizer & Loss:** Add standard optimization algorithms (e.g., SGD, Adam) and a loss function (e.g., Cross-Entropy) using NumPy.
3. **Develop NumPy Training Loop:** Create a script to handle data loading, batching, forward/backward pass, and weight updates using the pure NumPy components.
4. **Train and Evaluate (NumPy):** Train the NumPy model on the identity and reverse string tasks and evaluate its performance.
5. **Refactor and Port to JAX:**
    - Refactor the existing NumPy codebase to be compatible with JAX (shift from OOP, module-based paradigm to rather functional paradigm, if necessary).
    - Leverage `jax.jit` to accelerate computations significantly.
    - Replace the manual backward pass implementation with JAX's automatic differentiation (`jax.grad`) for potentially simpler and more robust gradient calculations.
    - Compare the performance and development experience of the JAX version against the NumPy version.
6. **Hyperparameter Tuning:** Experiment with different model sizes, learning rates, etc. (applicable to both NumPy and JAX versions).

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Transformer from Scratch (PyTorch)](https://www.kaggle.com/code/aisuko/transformer-from-scratch/notebook)
- [Let's code a Neural Network in plain NumPy](https://medium.com/data-science/lets-code-a-neural-network-in-plain-numpy-ae7e74410795)

## Project Members

- [@benitomano](https://github.com/benitomano)
- [@chanagel](https://github.com/chanagel)
- [@le0ff](https://github.com/le0ff)
- [@lorena-schl](https://github.com/lorena-schl)
- [@Mangyoo](https://github.com/Mangyoo)