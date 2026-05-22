# LoRA from Scratch on GPT-2 

This project implements **Low-Rank Adaptation (LoRA)** from scratch using PyTorch and applies it to a frozen GPT-2 (124M) model. The entire workflow is implemented in a **Jupyter Notebook (`Padma_lora.ipynb`)**, focusing on understanding and demonstrating parameter-efficient fine-tuning (PEFT).

---

## 🚀 Project Highlights

- Implemented **LoRA (Low-Rank Adaptation)** from scratch using PyTorch
- Injected LoRA layers into a frozen **GPT-2 small (124M)** model
- Full training pipeline implemented in **Jupyter Notebook**
- Fine-tuned on **TinyShakespeare dataset**
- Verified only LoRA parameters are trainable
- Generated Shakespeare-style text
- Compared conceptually with Hugging Face **PEFT library**
- Saved lightweight adapter weights instead of full model

---

## 🧠 Notebook

All experiments and training are done inside:
Padma_lora.ipynb

This notebook includes:
- Model loading (GPT-2)
- LoRA layer implementation
- Dataset preprocessing (TinyShakespeare)
- Training loop
- Text generation
- Evaluation experiments

---

## 📊 Dataset

- **TinyShakespeare**
- Character-level dataset
- Used for learning Shakespeare-style language modeling

---

## 🏗️ Project Structure

```text
Projects/
│
├── Padma_lora.ipynb    # Main implementation notebook
│
├── README.md                  # Project documentation
│
└── fix_notebook.py     # Utility to fix GitHub rendering issues
│
└──Requirements.txt

```

---

## ⚙️ Setup Instructions

```bash id="setup1"
git clone https://github.com/your-username/lora-gpt2.git
cd lora-gpt2
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

```
## 🚀 How to Run

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open the notebook:

```bash
Padma_lora.ipynb
```

3. Run all cells sequentially to execute:
- LoRA implementation
- GPT-2 fine-tuning
- Training loop
- Text generation

---

## 💾 Output

After fine-tuning, the model generates pseudo-Shakespearean text in the style of TinyShakespeare.

### Example Output

```text
ROMEO:
O gentle night, thou art more fair than dawn...

HAMLET:
To be, or not to be, that is the question...
```

---

## 🔁 PEFT Comparison

The notebook also includes a conceptual comparison between the custom LoRA implementation and the Hugging Face `peft` library implementation.

This helps validate the correctness of the custom LoRA approach and demonstrates understanding of parameter-efficient fine-tuning techniques.

---

## 🧪 Key Learnings

- Understanding LoRA and low-rank matrix decomposition
- Fine-tuning transformer models efficiently
- Freezing base model weights while training adapters
- Internal architecture of GPT-2
- Parameter-efficient fine-tuning (PEFT)
- Practical experience with LLM adaptation workflows

---

## 🧰 Tech Stack

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Jupyter Notebook
- PEFT

---

## 📌 Future Improvements

- Convert notebook workflow into a production-ready `train.py`
- Add evaluation metrics such as perplexity and loss curves
- Experiment with different style-transfer datasets
  - Yoda quotes
  - Dialogue datasets
  - Character-specific text corpora
- Deploy inference API for text generation
- Add attention-only LoRA injection