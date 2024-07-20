# ZKSL: Zero-Knowledge Split Learning

## TODO
- [ ] Plonky2 FRI for polynomial commitment
- [ ] Poseidon hash function
- [ ] Sha256 hash function
- [ ] Lassio for Plonk
- [ ] Folding for Plonk

## Overview

ZKSL (Zero-Knowledge Split Learning) is a project aimed at leveraging zero-knowledge proofs to verify the integrity and correctness of the split learning process in machine learning. With the increasing prevalence of on-device machine learning and collaborative model training between edge devices and servers, there is a growing need for methods that can both enhance privacy and ensure the validity of the training process. ZKSL addresses these needs by providing a zero-knowledge proof-based solution to validate the entire machine learning process.

## Background

The rise of on-device machine learning and its integration with server-side models has led to the adoption of split learning techniques. Split learning allows the model to be partitioned such that an embedding layer is trained on the device, and only this embedding is sent to the server for downstream tasks. This method helps in preserving user privacy by preventing the server from accessing raw user data.

However, this approach raises several critical questions:
- How can we ensure that the edge device has used the intended model for inference?
- How can we verify that the model parameters have been correctly updated during training?

Zero-knowledge proofs (ZKPs) offer a promising solution to these verification challenges. By incorporating ZKPs into the split learning process, we can ensure the integrity and correctness of the training without revealing any sensitive data.

## Project Structure

The ZKSL project is structured into three main stages, each representing a critical phase of the machine learning process. Each stage has its own zero-knowledge learning circuit to ensure end-to-end verification:

1. **Forward Stage**:
    - The forward propagation step involves calculating the outputs from the input data through the model layers.
    - This stage's zero-knowledge circuit ensures that the forward propagation is performed correctly without revealing the actual data or intermediate computations.

2. **Gradient Calculation Stage**:
    - In this stage, the gradients and loss are calculated based on the forward propagation results and the true labels.
    - The zero-knowledge circuit for this stage verifies the correctness of the gradient calculations and loss computation.

3. **Backward Stage**:
    - The backward propagation step updates the model parameters based on the gradients calculated in the previous stage.
    - The zero-knowledge circuit for this stage ensures that the parameter updates are performed correctly, thus maintaining the integrity of the model training process.

## Tasks

### Numerics

- [x] Relu
- [x] Exp
- [ ] Ln
- [x] Max
- [x] Accumulator
- [x] Add
- [x] Sub
- [x] Mul
- [x] Div
- [x] Dot

### Operations

- [x] GEMM
- [ ] Conv
- [ ] Concat
- [x] ReLU
- [x] Softmax

### Loss
- [x] Softmax
- [ ] MSE

### Commitments
- [x] Poseidon
- [ ] Sha256

### Provers
- [x] KZG
- [ ] IPA

## ONNX Model Integration
The project uses ONNX (Open Neural Network Exchange) models as the standard input. ONNX is an open format built to represent machine learning models. It enables interoperability between different machine learning frameworks, allowing models to be trained in one framework and then transferred to another for inference. This flexibility makes ONNX an ideal choice for ZKSL.

Each ONNX operation (op) corresponds to an operation in ZKSL/operations. This ensures that the operations defined in the ONNX model are correctly mapped and processed by the ZKSL system, maintaining consistency and correctness throughout the machine learning process.

## Getting Started

To get started with ZKSL, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yeexiaozheng/zksl.git
    cd zksl
    ```

2. **Dependencies**:
    - Install Rust: Follow the instructions on the [Rust website](https://www.rust-lang.org/tools/install) to install Rust on your system.
    - Install Python: Ensure that you have Python 3.6 or higher installed on your system.

3. **Run the Examples**:
    - Detailed examples for each stage can be found in the `examples` directory.
    - Follow the instructions in the examples to understand how each stage and its corresponding zero-knowledge proof circuit works.

## Contributing

We welcome contributions to ZKSL! If you would like to contribute, please fork the repository and create a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## License

ZKSL is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions or further information, please open an issue on the GitHub repository or contact the project maintainers.
