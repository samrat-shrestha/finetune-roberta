"""
finetune.py
-----------
Fine-tunes obi/deid_roberta_i2b2 on JSONL data produced by generate_synthetic_data.py
(or on real i2b2/MIMIC-IV data in the same format).

What this script teaches you:
  1. How tokenizer alignment works (word tokens → subword tokens)
  2. How BILOU label encoding works
  3. How HuggingFace Trainer handles NER
  4. How to swap in real data later with zero changes to the training logic

Usage:
  # Quick smoke test (fast, CPU-friendly, 10 notes)
  python finetune.py --train data/synthetic_train.jsonl \
                     --val   data/synthetic_val.jsonl   \
                     --epochs 1 --batch_size 2 --max_samples 10 --output_dir ./test_run

  # Full training run
  python finetune.py --train data/synthetic_train.jsonl \
                     --val   data/synthetic_val.jsonl   \
                     --epochs 3 --batch_size 8 --output_dir ./runs/synthetic_v1

  # With LoRA (use when GPU VRAM < 16GB)
  python finetune.py --train data/synthetic_train.jsonl \
                     --val   data/synthetic_val.jsonl   \
                     --use_lora --epochs 3 --output_dir ./runs/synthetic_lora_v1

Requirements:
  pip install transformers datasets seqeval accelerate torch
  pip install peft          # only needed for --use_lora
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL = "obi/deid_roberta_i2b2"

# BILOU label set — must match OBI's label scheme exactly
# B = Beginning, I = Inside, L = Last token of multi-token entity,
# U = Unit (single-token entity), O = Outside
PHI_CATEGORIES = [
    "PATIENT", "STAFF", "DATE", "LOC", "HOSP",
    "AGE", "ID", "PHONE", "PATORG", "EMAIL", "OTHERPHI",
]

# Build full label list: O first, then B/I/L/U for each category
LABEL_LIST = ["O"]
for cat in PHI_CATEGORIES:
    LABEL_LIST += [f"B-{cat}", f"I-{cat}", f"L-{cat}", f"U-{cat}"]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Step 1: Load JSONL data
# ---------------------------------------------------------------------------

def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if max_samples:
        records = records[:max_samples]
    logger.info(f"Loaded {len(records)} records from {path}")
    return records


# ---------------------------------------------------------------------------
# Step 2: Convert character-level spans → word-level BILOU tags
#
# This is the trickiest part of NER fine-tuning. Here's why:
#
#   Input text:    "Patient John Smith was seen"
#   Character spans: [{start:8, end:18, label:"PATIENT"}]  →  "John Smith"
#
#   After word splitting: ["Patient", "John", "Smith", "was", "seen"]
#   Word-level BILOU:     [O,        B-PATIENT, L-PATIENT, O,    O  ]
#
#   After RoBERTa subword tokenization:
#   Tokens:    [<s>, "Pat", "ient", "John", "Sm", "ith", "was", "seen", </s>]
#   We need to:
#     - Assign the word's label to its FIRST subword token
#     - Assign -100 (ignored by loss) to all continuation subword tokens
#     - Assign -100 to special tokens [CLS]/[SEP]
# ---------------------------------------------------------------------------

def char_spans_to_word_bilou(text: str, spans: List[Dict]) -> List[Tuple[str, str]]:
    """
    Convert character-level PHI spans to a list of (word, bilou_label) pairs.
    
    Returns: list of (word_string, bilou_label) for every whitespace-split word.
    """
    # Build a character → span_index mapping for fast lookup
    char_to_span = {}
    for span_idx, span in enumerate(spans):
        for char_idx in range(span["start"], span["end"]):
            char_to_span[char_idx] = span_idx

    # Walk through words, tracking character offsets manually
    words_and_labels = []
    char_pos = 0
    # We split on whitespace but preserve the positions
    import re
    for match in re.finditer(r'\S+', text):
        word = match.group()
        word_start = match.start()
        word_end = match.end()

        # Find which span(s) this word overlaps
        overlapping_span_indices = set()
        for c in range(word_start, word_end):
            if c in char_to_span:
                overlapping_span_indices.add(char_to_span[c])

        if not overlapping_span_indices:
            words_and_labels.append((word, "O"))
        else:
            # Use the first (leftmost) overlapping span
            span_idx = min(overlapping_span_indices)
            span = spans[span_idx]
            label = span["label"]

            # Determine BILOU tag
            # A word is "U" if the entire span is within this single word
            # A word is "B" if it starts at/after span start but span continues beyond this word
            # A word is "L" if span started before this word but ends here
            # A word is "I" if span started before and continues after
            span_start = span["start"]
            span_end   = span["end"]

            is_first = word_start <= span_start < word_end or word_start == span_start
            is_last  = word_start < span_end <= word_end  or word_end == span_end

            # Refine: check if any previous word was already in this span
            already_started = any(
                (w, l) for (w, l) in words_and_labels
                if l in [f"B-{label}", f"I-{label}", f"L-{label}"]
                # crude check — works because spans are sorted
            )
            # Better check: look at the span directly
            prev_word_in_span = word_start > span_start

            if prev_word_in_span:
                # Span started before this word
                if word_end >= span_end:
                    bilou = f"L-{label}"   # last word of span
                else:
                    bilou = f"I-{label}"   # inside span
            else:
                # This word starts at or after span start
                if word_end >= span_end:
                    bilou = f"U-{label}"   # entire span is this word
                else:
                    bilou = f"B-{label}"   # beginning of multi-word span

            words_and_labels.append((word, bilou))

    return words_and_labels


# ---------------------------------------------------------------------------
# Step 3: Tokenize and align labels
#
# HuggingFace tokenizers produce `word_ids()` — a list that maps each
# subword token back to its original word index. We use this to:
#   - Copy the word's BILOU label to the first subword token of that word
#   - Set -100 for all subsequent subword tokens of the same word
#   - Set -100 for special tokens
# ---------------------------------------------------------------------------

def tokenize_and_align_labels(examples, tokenizer, max_length=512):
    """
    Takes a batch of examples with 'words' and 'bilou_labels' fields.
    Returns tokenized inputs with aligned 'labels' field.
    """
    tokenized = tokenizer(
        examples["words"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,   # ← tells tokenizer input is pre-split into words
    )

    all_labels = []
    for i, word_labels in enumerate(examples["bilou_labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special token [CLS] / [SEP] — ignored in loss
                label_ids.append(-100)
            elif word_id != prev_word_id:
                # First subword of a new word — assign the word's label
                label_ids.append(LABEL2ID[word_labels[word_id]])
            else:
                # Continuation subword — ignored in loss
                # (Some practitioners assign the same label; -100 is the standard)
                label_ids.append(-100)
            prev_word_id = word_id

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


# ---------------------------------------------------------------------------
# Step 4: Metrics — seqeval-based precision/recall/F1 + FN per 1000 tokens
# ---------------------------------------------------------------------------

def make_compute_metrics(label_list):
    """Returns a compute_metrics function for HuggingFace Trainer."""
    try:
        from seqeval.metrics import (
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )
        from seqeval.scheme import BILOU
        USE_SEQEVAL = True
    except ImportError:
        logger.warning("seqeval not installed — using simple token-level accuracy instead")
        USE_SEQEVAL = False

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove -100 (padding / subword continuations)
        true_labels  = [[label_list[l] for l in label if l != -100]
                        for label in labels]
        true_preds   = [[label_list[p] for p, l in zip(pred, label) if l != -100]
                        for pred, label in zip(predictions, labels)]

        if USE_SEQEVAL:
            results = {
                "precision": precision_score(true_labels, true_preds, scheme=BILOU, zero_division=0),
                "recall":    recall_score(true_labels, true_preds,    scheme=BILOU, zero_division=0),
                "f1":        f1_score(true_labels, true_preds,        scheme=BILOU, zero_division=0),
            }
        else:
            # Fallback: flat token accuracy (not great for NER but works)
            flat_true = [l for seq in true_labels for l in seq]
            flat_pred = [p for seq in true_preds  for p in seq]
            correct = sum(t == p for t, p in zip(flat_true, flat_pred))
            results = {"accuracy": correct / len(flat_true)}

        # --- Sensitivity metric: FN per 1000 tokens ---
        # A False Negative is a PHI token predicted as O
        fn_count = 0
        total_tokens = 0
        for true_seq, pred_seq in zip(true_labels, true_preds):
            for t, p in zip(true_seq, pred_seq):
                total_tokens += 1
                if t != "O" and p == "O":
                    fn_count += 1
        results["fn_per_1000_tokens"] = (fn_count / max(total_tokens, 1)) * 1000

        return results

    return compute_metrics


# ---------------------------------------------------------------------------
# Step 5: Main training loop
# ---------------------------------------------------------------------------

def prepare_dataset(records, tokenizer, max_length=512):
    """Convert raw JSONL records → HuggingFace Dataset ready for Trainer."""

    all_words  = []
    all_labels = []

    for record in records:
        text  = record["text"]
        spans = record["spans"]

        word_label_pairs = char_spans_to_word_bilou(text, spans)
        words  = [w for w, _ in word_label_pairs]
        labels = [l for _, l in word_label_pairs]

        all_words.append(words)
        all_labels.append(labels)

    raw_dataset = Dataset.from_dict({
        "words":        all_words,
        "bilou_labels": all_labels,
    })

    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["words", "bilou_labels"],
    )

    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",       required=True,  help="Path to training JSONL")
    parser.add_argument("--val",         required=True,  help="Path to validation JSONL")
    parser.add_argument("--output_dir",  default="./runs/finetune_v1")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=2e-5,
                        help="Learning rate (Stanford CS224N suggests 2e-5 for BERT-family)")
    parser.add_argument("--max_length",  type=int,   default=512)
    parser.add_argument("--max_samples", type=int,   default=None,
                        help="Cap dataset size (useful for smoke tests)")
    parser.add_argument("--use_lora",    action="store_true",
                        help="Use LoRA for parameter-efficient fine-tuning (needs: pip install peft)")
    parser.add_argument("--warmup_ratio",type=float, default=0.1)
    parser.add_argument("--weight_decay",type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Label set ({len(LABEL_LIST)} labels): {LABEL_LIST[:8]}...")

    # --- Load tokenizer ---
    logger.info(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # --- Load and prepare data ---
    train_records = load_jsonl(args.train, args.max_samples)
    val_records   = load_jsonl(args.val,   args.max_samples)

    logger.info("Converting character spans → BILOU tokens and tokenizing...")
    train_dataset = prepare_dataset(train_records, tokenizer, args.max_length)
    val_dataset   = prepare_dataset(val_records,   tokenizer, args.max_length)

    logger.info(f"Train: {len(train_dataset)} examples | Val: {len(val_dataset)} examples")

    # --- Load model ---
    logger.info(f"Loading model from {BASE_MODEL}...")
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # ← needed because we may change num_labels
    )

    # --- Optional: apply LoRA ---
    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            logger.info("Applying LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=16,                    # rank — higher = more capacity, more VRAM
                lora_alpha=32,           # scaling factor (rule of thumb: 2×r)
                lora_dropout=0.1,
                target_modules=["query", "value"],  # RoBERTa attention projections
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            logger.error("peft not installed. Run: pip install peft")
            raise

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Core hyperparameters
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=50,                 # ~10% of 500 samples @ batch 8 = 500 steps total

        # Evaluation and saving
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Logging
        logging_steps=10,
        report_to="none",               # swap to "wandb" when you're ready to track experiments

        # Reproducibility
        seed=42,

        # Speed
        fp16=torch.cuda.is_available(), # use half precision on GPU automatically
        dataloader_num_workers=0,       # set to 4 on a proper GPU machine
    )

    # --- Data collator (handles dynamic padding) ---
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(LABEL_LIST),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # --- Train ---
    logger.info("=" * 60)
    logger.info("Starting fine-tuning...")
    logger.info(f"  Model:       {BASE_MODEL}")
    logger.info(f"  Train size:  {len(train_dataset)}")
    logger.info(f"  Val size:    {len(val_dataset)}")
    logger.info(f"  Epochs:      {args.epochs}")
    logger.info(f"  Batch size:  {args.batch_size}")
    logger.info(f"  LR:          {args.lr}")
    logger.info(f"  LoRA:        {args.use_lora}")
    logger.info(f"  Device:      {'GPU' if torch.cuda.is_available() else 'CPU'}")
    logger.info("=" * 60)

    train_result = trainer.train()

    # --- Save ---
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # --- Final eval ---
    logger.info("Running final evaluation on validation set...")
    metrics = trainer.evaluate()

    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info(f"  {k:<30} {v:.4f}")

    # Save metrics to file for experiment tracking
    metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nMetrics saved to {metrics_path}")
    logger.info(f"Model saved to   {args.output_dir}")


if __name__ == "__main__":
    main()
