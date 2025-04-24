# Ava Language Model Project Overview

**Ava** is a custom-designed Transformer-based Causal Language Model (CLM) developed with full control over architecture, configuration, and training. It is intended for experimentation, educational use, and lightweight deployments, especially in resource-constrained environments.

---

## Project Purpose

The goal of Ava is to enable:

- Full-stack development of a language model from scratch
- Deep understanding of transformer internals
- Flexible fine-tuning on local or personal datasets
- Easy customization and modular expansion

Ava is suitable for researchers, developers, and hobbyists interested in building LLMs without the constraints of large-scale frameworks.

---

## Architecture Highlights

- **Transformer Decoder Stack**: Implements multi-head attention, feedforward networks, and residual connections.
- **Rotary Positional Embeddings**: Enhances context handling without fixed positional encodings.
- **Flexible Configurations**: Supports model sizes from 100M to 100B parameters via `AvaConfig`.
- **LoRA (Low-Rank Adaptation)**: Enables parameter-efficient fine-tuning.
- **Quantization Support**: Reduces model size for low-memory inference.
- **Custom Dataset Handling**: Optimized for conversational and pretraining datasets.

---

## Training Pipeline

Ava includes a complete training workflow:

1. **Data Preparation**: Conversational data is loaded from JSON and tokenized.
2. **Model Configuration**: Users choose a predefined size (`100m`, `1b`, `7b`, etc.) or customize one.
3. **Training Loop**: Modular trainer with validation, checkpointing, and optional evaluation.
4. **Evaluation & Generation**: Supports text generation with temperature, top-k, and top-p sampling.

---

## Customization Options

- **Tokenizer**: Plug in your own tokenizer (e.g., BPE for custom languages).
- **LoRA Fine-Tuning**: Target specific layers to update with LoRA.
- **Model Quantization**: Use 8-bit weights for faster inference on CPUs.
- **Streaming Support**: Integrate with streamers for interactive generation.

---

## Use Cases

- Local AI assistants
- Chatbots for under-resourced languages
- Educational demos and research
- Small-scale AGI experiments
- Edge and offline deployments

---

## How to Get Started

To train Ava, users need:

- PyTorch environment with GPU (or CPU for smaller models)
- JSON-formatted dataset
- Pretrained tokenizer (or train your own)
- Training script using the provided trainer module

The model outputs checkpoints and can be resumed or evaluated at any point.

---

## License & Contributions

Ava is intended for educational and personal use. Contributions are welcome to enhance architecture, training stability, and downstream applications.

---

## Citations

- [Vaswani et al. (2017). *Attention is All You Need.*](https://arxiv.org/abs/1706.03762)
- [Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*](https://arxiv.org/abs/2106.09685)
- [Press et al. (2021). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.*](https://arxiv.org/abs/2108.12409)

---

## Author & Credits

Created and maintained by Nika Kudukhashvili. Ava represents an ongoing project in building fully independent and explainable language models.

For questions or contributions, visit: [Github](https://github.com/Kuduxaaa)
