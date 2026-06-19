# NLP Project: Positive Review Generation and Machine Translation

This project investigates two significant applications of Natural Language Processing (NLP): sentiment-controlled text generation and English-to-Vietnamese machine translation. The work combines Reinforcement Learning from Human Feedback (RLHF) with modern Neural Machine Translation (NMT) techniques to demonstrate both language generation and translation capabilities.

---

## 1. Positive Review Generation Using RLHF

This module focuses on generating positive product reviews by fine-tuning a pretrained language model through Reinforcement Learning. The objective is to guide the model toward producing coherent and sentiment-aware text.

### Model Overview

* **Pretrained Model:** `distilgpt2`
* **Optimization Method:** Proximal Policy Optimization (PPO)
* **Framework:** Actor-Critic Reinforcement Learning
* **Value Estimator:** Multi-Layer Perceptron (MLP) with ReLU activation

### Training Configuration

| Hyperparameter              | Value |
| --------------------------- | ----- |
| Clip Ratio                  | 0.2   |
| Value Coefficient           | 0.5   |
| Entropy Coefficient         | 0.01  |
| Policy Learning Rate        | 3e-5  |
| Value Network Learning Rate | 9e-5  |

### Reward Mechanism

The reward function evaluates generated text based on three criteria:

1. **Positive Keyword Coverage**
   Counts the occurrence of predefined positive expressions within the generated review.

2. **Sentiment Similarity**
   Measures cosine similarity between the TF-IDF representation of the generated text and a positive sentiment reference vector.

3. **Length Encouragement**
   Rewards longer and more informative responses.

The final reward is normalized using the following formula:

[
R_{final} = \frac{\tanh(1.5R)}{2} + 0.5
]

This transformation stabilizes the reward range and improves training efficiency.

---

## 2. English-to-Vietnamese Machine Translation

This section explores two different translation approaches for converting English text into Vietnamese.

### Approach 1: Transformer Architecture from Scratch

A complete Transformer model inspired by the architecture proposed by Vaswani et al. is implemented using TensorFlow/Keras.

#### Main Components

* **Subword Tokenization:** Byte Pair Encoding (BPE) implemented through SentencePiece.
* **Multi-Head Attention:** Captures contextual information from multiple representation spaces.
* **Positional Encoding:** Injects sequence-order information using sinusoidal functions.
* **Encoder-Decoder Structure:** Utilizes residual connections and layer normalization to improve convergence and stability.

### Approach 2: Fine-Tuning MarianMT

This approach leverages the pretrained `Helsinki-NLP/opus-mt-en-vi` translation model and further fine-tunes it on conversational English-Vietnamese sentence pairs.

The goal is to enhance translation quality, fluency, and contextual accuracy for commonly used expressions and everyday communication.

---

## Installation

### Requirements

* Python 3.8 or higher
* PyTorch
* TensorFlow
* Hugging Face Transformers
* SentencePiece
* Scikit-learn
* NumPy
* Pandas

Install the required packages:

```bash
pip install torch tensorflow transformers sentencepiece scikit-learn numpy pandas
```

---

## Project Execution

The complete workflow, including data preprocessing, model training, inference, and evaluation, can be found in:

```text
source/source.ipynb
```

---

## Project Structure

```text
.
├── dataset/
│   ├── train.en.txt
│   └── train.vi.txt
│
├── source/
│   └── source.ipynb
│
└── README.md
```

---

## Notes

* Ensure that the notebook's working directory is configured correctly before execution.
* When loading datasets, use relative paths whenever possible to improve portability across different environments.
* Depending on available hardware resources, training time may vary significantly for both the RLHF and translation modules.
