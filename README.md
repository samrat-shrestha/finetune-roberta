# DeIdentiPhi

Fine-tuning [obi/deid_roberta_i2b2](https://huggingface.co/obi/deid_roberta_i2b2) for clinical text de-identification with an error-driven retraining loop.

## What This Project Does

1. **Generates synthetic clinical notes** with labeled PHI spans
2. **Fine-tunes a RoBERTa NER model** to detect and tag 11 categories of PHI
3. **Evaluates** the fine-tuned model against the baseline
4. **Iteratively improves** the model using an error-driven retraining loop

### PHI Categories Detected

| Category | Examples |
|---|---|
| `PATIENT` | John Smith, Maria Garcia |
| `STAFF` | Dr. Johnson, Dr. Lee |
| `DATE` | 03/12/1989, March 12, 1989 |
| `LOC` | 1234 Main St, New Orleans, LA |
| `HOSP` | Tulane Medical Center |
| `AGE` | 72, 45 |
| `ID` | MRN1234567, PT-9876543 |
| `PHONE` | 504-555-1234 |
| `EMAIL` | john.smith@gmail.com |
| `PATORG` | Patient organizations |
| `OTHERPHI` | Any other identifying info |

---

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.x (CUDA recommended for GPU acceleration)
- HuggingFace Transformers, Datasets, seqeval
- Optional: `peft` for LoRA fine-tuning (`pip install peft`)

---

## Project Structure

```
DeIdentiPhi/
├── generate_synthetic_data.py   # Step 1: Generate synthetic training data
├── finetune.py                  # Step 2: Fine-tune the model
├── baseline_eval.py             # Step 3: Evaluate baseline vs fine-tuned
├── error_analysis.py            # Step 4: Per-category error breakdown
├── loop_retrain.py              # Step 5: Error-driven retraining loop
├── data/
│   ├── synthetic_train.jsonl    # Generated training data (500 notes)
│   └── synthetic_val.jsonl      # Generated validation data (100 notes)
├── runs/                        # Model checkpoints and training outputs
├── requirements.txt
└── README.md
```

---

## Quick Start

### Step 1: Generate Synthetic Data

```bash
# Training set (500 notes)
python generate_synthetic_data.py --num_notes 500 --output data/synthetic_train.jsonl --seed 42

# Validation set (100 notes)
python generate_synthetic_data.py --num_notes 100 --output data/synthetic_val.jsonl --seed 99
```

Each note is a realistic clinical document (discharge summary, progress note, referral letter, or ED note) with character-level PHI span annotations.

### Step 2: Fine-Tune the Model

```bash
# Full training run (GPU recommended)
python finetune.py --train data/synthetic_train.jsonl \
                   --val data/synthetic_val.jsonl \
                   --epochs 3 --batch_size 8 \
                   --output_dir ./runs/synthetic_v1
```

**Quick smoke test** (CPU-friendly, ~2 min):
```bash
python finetune.py --train data/synthetic_train.jsonl \
                   --val data/synthetic_val.jsonl \
                   --epochs 1 --batch_size 2 --max_samples 10 \
                   --output_dir ./runs/test_run
```

**With LoRA** (for GPUs with < 16GB VRAM):
```bash
python finetune.py --train data/synthetic_train.jsonl \
                   --val data/synthetic_val.jsonl \
                   --use_lora --epochs 3 \
                   --output_dir ./runs/synthetic_lora_v1
```

### Step 3: Evaluate Against Baseline

```bash
# Baseline only
python baseline_eval.py --val data/synthetic_val.jsonl

# Baseline vs fine-tuned comparison
python baseline_eval.py --val data/synthetic_val.jsonl \
                        --finetuned_model ./runs/synthetic_v1
```

### Step 4: Error Analysis (Optional)

```bash
python error_analysis.py --model ./runs/synthetic_v1 \
                         --val data/synthetic_val.jsonl \
                         --output error_report.json
```

Produces a per-category breakdown showing which PHI types the model struggles with most.

### Step 5: Error-Driven Retraining Loop

```bash
# Full loop (GPU recommended)
python loop_retrain.py --train data/synthetic_train.jsonl \
                       --val data/synthetic_val.jsonl \
                       --max_iterations 5 \
                       --fn_target 0.3 \
                       --epochs 3 --batch_size 8 \
                       --output_dir ./runs/loop
```

**Quick smoke test:**
```bash
python loop_retrain.py --train data/synthetic_train.jsonl \
                       --val data/synthetic_val.jsonl \
                       --max_iterations 2 --max_samples 50 \
                       --epochs 2 --batch_size 4 \
                       --output_dir ./runs/loop_test
```

---

## How the Retraining Loop Works

The loop iteratively improves the model by analyzing its weaknesses and generating targeted training data:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Train Model ──► Analyze Errors ──► Target Met?            │
│       ▲                                  │                  │
│       │                            Yes: Stop ✅             │
│       │                            No:  ▼                   │
│       │                                                     │
│       └── Merge Data ◄── Generate Targeted Data             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Each iteration:**
1. **Train** — Fine-tune the model on the current training data
2. **Analyze** — Run the model on validation data, count false negatives per PHI category
3. **Check** — Exit if FN/1000 ≤ target AND F1 ≥ 0.8, or if metrics plateau
4. **Generate** — Create targeted synthetic data emphasizing weak categories
5. **Merge** — Combine original + targeted data, repeat

**Exit conditions:**
- ✅ FN target reached with valid F1 score
- ⛔ Metrics plateau (no improvement between iterations)
- ⏱️ Maximum iterations reached

**Output structure:**
```
runs/loop/
├── iter_1/
│   ├── model/               # Saved model checkpoint
│   ├── error_report.json    # Per-category error analysis
│   ├── metrics.json         # Training + eval metrics
│   └── targeted_data.jsonl  # Generated targeted data
├── iter_2/
│   └── ...
├── best_model/              # Copy of the best iteration's model
└── summary.json             # Metrics trajectory across all iterations
```

---

## Results

### Baseline vs Fine-Tuned

| Metric | Baseline | Fine-Tuned | Improvement |
|---|---|---|---|
| Precision | 0.791 | 0.999 | +26.4% |
| Recall | 0.786 | 0.993 | +26.3% |
| F1 | 0.788 | 0.996 | +26.4% |
| FN/1000 tokens | 21.37 | 0.93 | -95.6% |

### Metrics Explained

- **Precision**: When the model flags a token as PHI, how often is it correct?
- **Recall**: Of all actual PHI tokens, what percentage did the model catch?
- **F1**: Harmonic mean of precision and recall (overall quality score)
- **FN/1000 tokens**: False negatives per 1000 tokens — the critical safety metric. Each false negative is a PHI token that would leak through de-identification. Lower is safer.

---

## CLI Reference

### `generate_synthetic_data.py`

| Argument | Default | Description |
|---|---|---|
| `--num_notes` | 500 | Number of notes to generate |
| `--output` | `data/synthetic_train.jsonl` | Output file path |
| `--seed` | 42 | Random seed |
| `--category_weights` | None | JSON string of category weights for targeted generation |

### `finetune.py`

| Argument | Default | Description |
|---|---|---|
| `--train` | required | Path to training JSONL |
| `--val` | required | Path to validation JSONL |
| `--output_dir` | `./runs/finetune_v1` | Output directory |
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 2e-5 | Learning rate |
| `--max_length` | 512 | Max token sequence length |
| `--max_samples` | None | Cap dataset size (for testing) |
| `--use_lora` | False | Use LoRA for parameter-efficient fine-tuning |
| `--warmup_ratio` | 0.1 | Warmup ratio |
| `--weight_decay` | 0.01 | Weight decay |

### `baseline_eval.py`

| Argument | Default | Description |
|---|---|---|
| `--val` | required | Path to validation JSONL |
| `--baseline_model` | `obi/deid_roberta_i2b2` | Baseline model ID |
| `--finetuned_model` | None | Path to fine-tuned model (optional) |
| `--max_samples` | None | Cap validation set size |

### `error_analysis.py`

| Argument | Default | Description |
|---|---|---|
| `--model` | required | Path to fine-tuned model |
| `--val` | required | Path to validation JSONL |
| `--output` | `error_report.json` | Output report path |
| `--max_samples` | None | Cap validation set size |

### `loop_retrain.py`

| Argument | Default | Description |
|---|---|---|
| `--train` | required | Path to training JSONL |
| `--val` | required | Path to validation JSONL |
| `--max_iterations` | 5 | Maximum loop iterations |
| `--fn_target` | 0.5 | Target FN/1000 tokens |
| `--f1_floor` | 0.8 | Minimum F1 for model validity |
| `--targeted_notes_per_iter` | 200 | Targeted notes generated per iteration |
| `--epochs` | 3 | Training epochs per iteration |
| `--batch_size` | 8 | Batch size |
| `--lr` | 2e-5 | Learning rate |
| `--max_samples` | None | Cap dataset size |
| `--use_lora` | False | Use LoRA |
| `--output_dir` | `./runs/loop` | Output directory |

---

## Base Model

This project fine-tunes [obi/deid_roberta_i2b2](https://huggingface.co/obi/deid_roberta_i2b2), a RoBERTa model pre-trained for clinical NER de-identification by the [Obi](https://github.com/obi-ml-public/ehr_deidentification) team. The model uses BILOU (Beginning, Inside, Last, Outside, Unit) label encoding for entity boundaries.
