Here is the explanation of the configuration parameters, formatted in **Markdown** to match the style of your previous `README` sections:

# 🧪 Configuration Parameters — Molecule Natural Language Training

This configuration sets up training for a BERT-style model on a dataset of molecular natural language descriptions. Below is an explanation of each parameter used:


---

## ⚙️ Training Hyperparameters

* **`max_steps = 600000`**
  Total number of training steps (i.e., optimizer updates).

* **`learning_rate = 6e-6`**
  Learning rate for the optimizer.

* **`weight_decay = 1e-1`**
  Weight decay for L2 regularization to reduce overfitting.

* **`log_interval = 100`**
  Log training metrics (e.g., loss) every 100 steps.

---

## 📦 Dataset & Model Paths

* **`dataset_dir`**
  Path to the preprocessed Hugging Face-compatible dataset:
    THIS IS THE PATH TO YOUR PROCESSED DATASET. See [Training of GPT2 model Section in](../README.md).

* **`model_path = "runs_train_bert_molecule_nl"`**
  Directory where model outputs (checkpoints, logs) will be saved.

---

## 🔠 Tokenization Parameters

* **`max_length = 1024`**
  Maximum sequence length for input tokenization. Inputs longer than this will be truncated.

---

## 📦 Batch Settings

* **`batch_size = 1`**
  Training batch size per GPU/device.

* **`per_device_eval_batch_size = 1`**
  Evaluation batch size per GPU/device.

* **`gradient_accumulation_steps = 5 * 16`**
  Number of steps to accumulate gradients before an optimizer step.
  Allows simulating a larger effective batch size without exceeding memory limits.

> 🧮 **Effective Batch Size** = `batch_size × gradient_accumulation_steps × num_GPUs`

---

## 🔖 Special Tokens

* **`start_instruction = 1`**
  Token ID representing the start of an instruction.

* **`end_instruction = [518, 29914, 25580, 29962]`**
  Token IDs marking the end of an instruction block.

* **`eos_token = 2`**
  Token ID for the end-of-sequence marker.

---

## 📏 Model Variant

* **`model_size = "small"`**
  Specifies the model variant to use: choose between `"small"`, `"medium"`, or `"large"` depending on available compute and use case.
