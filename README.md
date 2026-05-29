# LoRA from Scratch on GPT-2 

This project implements Low-Rank Adaptation (LoRA) from scratch using PyTorch and applies it to a frozen GPT-2 (124M) model for parameter-efficient fine-tuning (PEFT) experiments.

The implementation focuses on understanding and reproducing the core mechanics of LoRA, including low-rank adapter injection, frozen base-model training, and lightweight fine-tuning for transformer architectures.

---

## 🚀 Project Highlights

- Implemented LoRA layers manually using PyTorch
- Injected trainable low-rank adapters into a frozen GPT-2 model
- Fine-tuned GPT-2 on the TinyShakespeare dataset
- Verified that only LoRA adapter parameters were updated during training
- Generated Shakespeare-style text after fine-tuning
- Compared the custom implementation conceptually with Hugging Face PEFT

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

The project uses the TinyShakespeare dataset for lightweight language-model fine-tuning and text generation experiments.

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

3. Run the notebook sequentially to:

- initialize GPT-2
- inject LoRA adapters
- fine-tune the model
- generate text samples

---

## 📈 Results

- Successfully fine-tuned GPT-2 using LoRA adapters only
- Reduced trainable parameters compared to full fine-tuning
- Generated coherent Shakespeare-style text samples after training

---

## 🧪 Key Concepts Demonstrated

- Parameter-Efficient Fine-Tuning (PEFT)
- Low-rank matrix decomposition
- Transformer adaptation
- Frozen-weight fine-tuning
- GPT-2 architecture understanding
- Efficient LLM training workflows

---

## 📌 Future Improvements

- Move notebook logic into modular Python training scripts
- Add evaluation metrics such as perplexity and validation loss
- Support configurable LoRA injection targets
- Add experiment tracking and checkpointing

