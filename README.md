# triton-toolkit
TC3002B: Toolkits used for finetune LLMs to acomodate for Triton input. 

## Repository Structure

### General Structure
```
├── compilar
├── data-processing
├── finetuning
├── grammar
├── notebooks
│   └── googleCollab
├── scripts
└── validators
    └── results
```

### `compiler/`

Contains the core implementation and validation scripts used to compare generated Triton kernels against their PyTorch equivalents.

* **`comparador.py`** – Runs comparisons between Triton and PyTorch implementations, validating correctness and output consistency.
* **`pytorch_impl.py`** – Reference PyTorch implementations used as ground truth.
* **`triton_impl.py`** – Triton kernel implementations generated or evaluated by the project.

---

### `data-processing/`

Contains notebooks and utilities for dataset preparation and preprocessing.

* **`dataprocessing.ipynb`** – Data cleaning, formatting, and transformation steps used before training or evaluation.

---

### `finetuning/`

Resources related to model fine-tuning experiments.

* **`finetuning9B.ipynb`** – Notebook used to fine-tune a 9B-parameter model for Triton kernel generation.

---

### `grammar/`

Contains the grammar-constrained decoding resources, prompts, and testing utilities.

* **`triton_grammar.gbnf`** – GBNF grammar defining the valid structure of generated Triton kernels.
* **`system_prompt.txt`** – System prompt used during inference or evaluation.
* **`specific_unit_tests.py`** – Targeted unit tests for validating generated kernels.
* **`triton_bench_tests.py`** – Integration tests based on TritonBench tasks.
* **`tutorial_tests.py`** – Additional validation examples and tutorial-style tests.

---

### `notebooks/`

General-purpose notebooks used during experimentation and development.

* **`base_notebook.ipynb`** – Main experimentation notebook.
* **`googleCollab/`** – Colab-specific notebooks.

  * **`finetuneOllama8BUnsloth.ipynb`** – Fine-tuning workflow using Unsloth and an 8B model.
* **`README.md`** – Documentation for notebook usage.

---

### `scripts/`

Reusable notebooks and scripts for training and experimentation workflows.

* **`base_cd.ipynb`** – Grammar-constrained decoding workflow.
* **`base_finetune.ipynb`** – Base fine-tuning pipeline.
* **`readme.md`** – Documentation for the scripts directory.

---

### `validators/`

Utilities for evaluating model outputs and measuring generation quality.

* **`count_valid_triton_kernels.py`** – Counts valid Triton kernels in prediction files and reports summary statistics.
* **`results/`** – Stores prediction outputs from different models and training configurations.

#### `validators/results/`

Contains generated predictions from various baseline, fine-tuned, and grammar-constrained models, including:

* GPT-5.5
* Gemini
* Qwen (Base, SFT, Constrained Decoding, and combined variants)
* IBM Granite (Base, SFT, Constrained Decoding, and combined variants)

These files are used for benchmarking, validation, and comparative analysis.

---

### Root Files

* **`README.md`** – Main project documentation and usage instructions.

Overall, the repository provides the complete pipeline for grammar-constrained Triton kernel generation, including dataset preparation, model fine-tuning, constrained decoding, kernel validation, and benchmarking against reference PyTorch implementations.
