# LoRA from Scratch on GPT-2 (PEFT Implementation)

This project implements Low-Rank Adaptation (LoRA) from scratch using PyTorch and applies it to a frozen GPT-2 (124M) model for parameter-efficient fine-tuning (PEFT) experiments.

The goal of this project is not only to fine-tune a language model, but to reproduce and deeply understand LoRA mechanics, including:

- low-rank decomposition of weight updates
- freezing pretrained transformer weights
- injecting trainable adapters into attention and MLP layers
- evaluating trade-offs using perplexity and style transfer

The implementation is fully built in a Jupyter Notebook (Padma_lora.ipynb).

---

## 🚀 Key Contributions

- Implemented LoRA from scratch in PyTorch
- Converted GPT-2 Conv1D layers to standard Linear layers for clean LoRA injection
- Injected LoRA into:
     - attention projection (c_attn)
     - attention output projection (attn.c_proj)
     - MLP projection (mlp.c_proj)
- Verified only LoRA parameters are trainable (~0.65% of model)
- Matched results with Hugging Face PEFT implementation
- Built full training loop with:
     - cosine learning rate schedule
     - gradient clipping
     - validation evaluation
- Evaluated using:
     - Shakespeare perplexity
     - control corpus perplexity (Pride & Prejudice)
- Performed style transfer experiments (Shakespeare / Yoda / dialogue datasets)
- Saved and loaded lightweight LoRA adapters (~3 MB)

---

## 🧠 LoRA Formulation (Core Idea)

For a frozen linear layer:

W₀x + ΔWx

LoRA models the update as:

ΔW = (B × A) × scaling

Where:

- A ∈ ℝ^(r × in_features)
- B ∈ ℝ^(out_features × r)
- r ≪ hidden size (low-rank constraint)

Only A and B are trained, while W₀ remains frozen.

---

## 🔁 Architecture Overview

The workflow follows a parameter-efficient fine-tuning pipeline:

1. Load pretrained GPT-2 (124M)
2. Freeze base transformer weights
3. Inject low-rank LoRA adapters
4. Train only adapter parameters
5. Perform inference using adapted model

This reduces the number of trainable parameters while preserving the pretrained model weights.

---
## 🧰 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Jupyter Notebook

---
## 📊 Dataset

### Primary Dataset
- TinyShakespeare
- Character-level dataset for language modeling
- Used for learning structured Shakespeare-style generation
  
### Control Dataset
- Excerpt from Pride & Prejudice
- Used to measure catastrophic forgetting
---

## 🏗️ Project Structure

```text

lora-gpt2/
│
├── Padma_lora.ipynb           # Main implementation notebook
│
├── README.md                  # Project documentation
│
└── fix_notebook.py            # Utility to fix GitHub rendering issues
│
└── requirements.txt

```

---

## ⚙️ Setup

```bash 
git clone <repository-url>
cd lora-gpt2

python -m venv .venv
source .venv/Scripts/activate       # Windows Git Bash

pip install -r requirements.txt

```
---

## 🚀 How to Run

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open the notebook:

```bash
Padma_lora.ipynb
```

3. Run all cells sequentially:

- Load GPT-2
- Inject LoRA adapters
- Train on dataset
- Evaluate perplexity
- Generate text samples

---

## 📈 Results

### Perplexity (Shakespeare)
  - Before LoRA: ~93
  - After LoRA: ~41
    
### Control Corpus (Pride & Prejudice)
  - Before: ~16.9
  - After: ~18.5
    
Key Observation
LoRA significantly improves domain-specific performance while only slightly affecting general language modeling ability.

---

## 🔁 PEFT Comparison

A key part of this project is verifying that the manual LoRA implementation matches Hugging Face PEFT.

Results:

- Trainable parameters: identical (~811K)
- Performance: nearly identical perplexity curves
- Output quality: similar Shakespeare-style generation

## 🧪 Key Learnings

- Low-rank matrix factorization in deep learning
- Transformer architecture internals (GPT-2)
- Efficient fine-tuning vs full fine-tuning
- Trade-off between specialization and generalization
- Practical implementation of PEFT systems
- Adapter-based model design

---

## 💾 Adapter System

The project supports lightweight saving/loading:

- Only LoRA parameters are stored (~3 MB)
- Base GPT-2 remains unchanged
- Adapters can be reloaded into fresh models

---

## 📌 Future Improvements

- Modular training script (train.py)
- Support multiple target modules dynamically
- Add experiment tracking (W&B / MLflow)
- Extend to larger models (GPT-Neo / LLaMA-style blocks)
- Add attention-only vs full-layer LoRA comparisons

