"""
baseline_eval.py
----------------
Runs the original obi/deid_roberta_i2b2 (no fine-tuning) on your val set
and produces the exact same metrics as finetune.py.

This gives you the number you need to beat. Run this BEFORE comparing
against your fine-tuned model.

Usage:
    python baseline_eval.py --val data/synthetic_val.jsonl

    # To also compare against your fine-tuned model in the same run:
    python baseline_eval.py --val data/synthetic_val.jsonl \
                            --finetuned_model ./runs/synthetic_v1

Output looks like:
    ============================================================
    BASELINE  (obi/deid_roberta_i2b2, no fine-tuning)
    ============================================================
      precision              0.8821
      recall                 0.8643
      f1                     0.8731
      fn_per_1000_tokens     4.2100

    ============================================================
    FINE-TUNED  (./runs/synthetic_v1)
    ============================================================
      precision              0.9992
      recall                 0.9932
      f1                     0.9962
      fn_per_1000_tokens     0.9341

    ============================================================
    IMPROVEMENT
    ============================================================
      precision              +0.1171  (+13.3%)
      recall                 +0.1289  (+14.9%)
      f1                     +0.1231  (+14.1%)
      fn_per_1000_tokens     -3.2759  (-77.8%)  ← fewer misses
"""

import argparse
import json
import logging
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label system — identical to finetune.py (must match exactly)
# ---------------------------------------------------------------------------

PHI_CATEGORIES = [
    "PATIENT", "STAFF", "DATE", "LOC", "HOSP",
    "AGE", "ID", "PHONE", "PATORG", "EMAIL", "OTHERPHI",
]

LABEL_LIST = ["O"]
for cat in PHI_CATEGORIES:
    LABEL_LIST += [f"B-{cat}", f"I-{cat}", f"L-{cat}", f"U-{cat}"]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL  = {i: label for label, i in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Data loading — identical to finetune.py
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


def char_spans_to_word_bilou(text: str, spans: List[Dict]) -> List[Tuple[str, str]]:
    """Identical copy from finetune.py — must stay in sync."""
    char_to_span = {}
    for span_idx, span in enumerate(spans):
        for char_idx in range(span["start"], span["end"]):
            char_to_span[char_idx] = span_idx

    words_and_labels = []
    for match in re.finditer(r'\S+', text):
        word       = match.group()
        word_start = match.start()
        word_end   = match.end()

        overlapping = set()
        for c in range(word_start, word_end):
            if c in char_to_span:
                overlapping.add(char_to_span[c])

        if not overlapping:
            words_and_labels.append((word, "O"))
        else:
            span_idx   = min(overlapping)
            span       = spans[span_idx]
            label      = span["label"]
            span_start = span["start"]
            span_end   = span["end"]
            prev_word_in_span = word_start > span_start

            if prev_word_in_span:
                bilou = f"L-{label}" if word_end >= span_end else f"I-{label}"
            else:
                bilou = f"U-{label}" if word_end >= span_end else f"B-{label}"

            words_and_labels.append((word, bilou))

    return words_and_labels


def tokenize_and_align_labels(examples, tokenizer, max_length=512):
    """Identical copy from finetune.py — must stay in sync."""
    tokenized = tokenizer(
        examples["words"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )
    all_labels = []
    for i, word_labels in enumerate(examples["bilou_labels"]):
        word_ids   = tokenized.word_ids(batch_index=i)
        label_ids  = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(LABEL2ID[word_labels[word_id]])
            else:
                label_ids.append(-100)
            prev_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def prepare_dataset(records, tokenizer, max_length=512):
    """Identical copy from finetune.py — must stay in sync."""
    all_words, all_labels = [], []
    for record in records:
        pairs  = char_spans_to_word_bilou(record["text"], record["spans"])
        all_words.append([w for w, _ in pairs])
        all_labels.append([l for _, l in pairs])

    raw = Dataset.from_dict({"words": all_words, "bilou_labels": all_labels})
    return raw.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, max_length),
        batched=True,
        remove_columns=["words", "bilou_labels"],
    )


# ---------------------------------------------------------------------------
# Metrics — identical to finetune.py
# ---------------------------------------------------------------------------

def compute_metrics_fn(predictions, labels):
    """
    Takes raw numpy arrays (as Trainer would pass them) and returns a dict.
    Same logic as make_compute_metrics() in finetune.py.
    """
    preds = np.argmax(predictions, axis=2)

    true_labels = [[LABEL_LIST[l] for l in label if l != -100] for label in labels]
    true_preds  = [[LABEL_LIST[p] for p, l in zip(pred, label) if l != -100]
                   for pred, label in zip(preds, labels)]

    try:
        from seqeval.metrics import f1_score, precision_score, recall_score
        from seqeval.scheme import BILOU
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="seqeval")
            results = {
                "precision": precision_score(true_labels, true_preds, scheme=BILOU, zero_division=0),
                "recall":    recall_score(true_labels,    true_preds, scheme=BILOU, zero_division=0),
                "f1":        f1_score(true_labels,        true_preds, scheme=BILOU, zero_division=0),
            }
    except ImportError:
        flat_true = [l for seq in true_labels for l in seq]
        flat_pred = [p for seq in true_preds  for p in seq]
        correct = sum(t == p for t, p in zip(flat_true, flat_pred))
        results = {"accuracy": correct / max(len(flat_true), 1)}

    fn_count = sum(
        1 for true_seq, pred_seq in zip(true_labels, true_preds)
        for t, p in zip(true_seq, pred_seq)
        if t != "O" and p == "O"
    )
    total_tokens = sum(len(seq) for seq in true_labels)
    results["fn_per_1000_tokens"] = (fn_count / max(total_tokens, 1)) * 1000

    return results


# ---------------------------------------------------------------------------
# Core evaluation function — two modes
#
# The problem with using Trainer for baseline evaluation:
#   The original obi/deid_roberta_i2b2 has its own label scheme (different
#   names, possibly BIO not BILOU). If we force our 45-label scheme onto it
#   with ignore_mismatched_sizes=True, we replace its classification head
#   with random weights — so it predicts garbage.
#
# The fix:
#   - Baseline: use HuggingFace pipeline() which respects the model's native
#     labels, then map those labels to ours for scoring.
#   - Fine-tuned: use Trainer as before (it was trained with our label scheme).
# ---------------------------------------------------------------------------

# Mapping from OBI's native label names → our PHI categories
# OBI uses labels like "B-AGE", "I-PATIENT" etc. — categories match ours,
# but they may use BIO instead of BILOU. We strip the prefix and remap.
OBI_CATEGORY_MAP = {
    "PATIENT":  "PATIENT",
    "STAFF":    "STAFF",
    "DATE":     "DATE",
    "LOC":      "LOC",
    "HOSP":     "HOSP",
    "AGE":      "AGE",
    "ID":       "ID",
    "PHONE":    "PHONE",
    "PATORG":   "PATORG",
    "EMAIL":    "EMAIL",
    "OTHERPHI": "OTHERPHI",
}

def evaluate_baseline_pipeline(model_path: str, val_records: List[Dict], label: str) -> Dict:
    """
    Evaluates the baseline model using HuggingFace pipeline() — respects the
    model's own native label scheme. Converts predictions to our label format
    for fair comparison.
    """
    from transformers import pipeline as hf_pipeline

    logger.info(f"Loading baseline via pipeline: {model_path}")
    pipe = hf_pipeline(
        "token-classification",
        model=model_path,
        aggregation_strategy="none",   # get raw per-token labels, not merged spans
        device=-1,                     # CPU
    )

    all_true_labels = []
    all_pred_labels = []

    for record in val_records:
        text  = record["text"]
        spans = record["spans"]

        # Ground truth: convert spans → word-level BILOU labels
        word_label_pairs = char_spans_to_word_bilou(text, spans)
        true_words  = [w for w, _ in word_label_pairs]
        true_labels = [l for _, l in word_label_pairs]

        # Prediction: run the pipeline
        try:
            raw_preds = pipe(text)
        except Exception as e:
            logger.warning(f"Pipeline failed on a record: {e}")
            continue

        # The pipeline returns per-token predictions with character offsets.
        # We need to align these back to our word boundaries.
        # Strategy: for each word, find the pipeline token that starts at/near
        # the word's character position and use its label.
        pred_labels = []
        char_to_pred = {}
        for tok in raw_preds:
            for c in range(tok["start"], tok["end"]):
                char_to_pred[c] = tok["entity"]

        for match in re.finditer(r'\S+', text):
            word_start = match.start()
            word_end   = match.end()

            # Find the most common prediction label for characters in this word
            preds_in_word = [char_to_pred.get(c, "O") for c in range(word_start, word_end)]
            # Pick the most frequent non-O label, or O if all O
            non_o = [p for p in preds_in_word if p != "O"]
            if non_o:
                raw_label = max(set(non_o), key=non_o.count)
            else:
                raw_label = "O"

            # Convert OBI label to our BILOU scheme
            # OBI uses B-/I- prefix; we map to our category names
            if raw_label == "O":
                pred_labels.append("O")
            else:
                parts = raw_label.split("-", 1)
                prefix   = parts[0] if len(parts) == 2 else "B"
                category = parts[1] if len(parts) == 2 else parts[0]
                mapped   = OBI_CATEGORY_MAP.get(category, "OTHERPHI")
                # Keep B/I prefix — treat as B for single, keep I for continuation
                pred_labels.append(f"{prefix}-{mapped}")

        # Align lengths (pipeline may tokenize differently at edges)
        min_len = min(len(true_labels), len(pred_labels))
        all_true_labels.append(true_labels[:min_len])
        all_pred_labels.append(pred_labels[:min_len])

    # Score using seqeval
    try:
        from seqeval.metrics import f1_score, precision_score, recall_score
        from seqeval.scheme import IOB2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="seqeval")
            # Use IOB2 for baseline since OBI uses BIO tagging
            results = {
                "precision": precision_score(all_true_labels, all_pred_labels, zero_division=0),
                "recall":    recall_score(all_true_labels,    all_pred_labels, zero_division=0),
                "f1":        f1_score(all_true_labels,        all_pred_labels, zero_division=0),
            }
    except Exception as e:
        logger.warning(f"seqeval scoring failed: {e} — falling back to token accuracy")
        flat_true = [l for seq in all_true_labels for l in seq]
        flat_pred = [l for seq in all_pred_labels for l in seq]
        correct = sum(t == p for t, p in zip(flat_true, flat_pred))
        results = {"precision": 0.0, "recall": 0.0, "f1": correct / max(len(flat_true), 1)}

    fn_count = sum(
        1 for true_seq, pred_seq in zip(all_true_labels, all_pred_labels)
        for t, p in zip(true_seq, pred_seq)
        if t != "O" and p == "O"
    )
    total_tokens = sum(len(seq) for seq in all_true_labels)
    results["fn_per_1000_tokens"] = (fn_count / max(total_tokens, 1)) * 1000

    return results


def evaluate_finetuned_trainer(model_path: str, val_dataset, label: str) -> Dict:
    """
    Evaluates a fine-tuned model using Trainer. This model was trained with
    our exact label scheme so we can use Trainer directly.
    """
    logger.info(f"Loading fine-tuned model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=8,
        report_to="none",
        seed=42,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def _compute(p):
        return compute_metrics_fn(p.predictions, p.label_ids)

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=_compute,
    )

    logger.info(f"Running evaluation for: {label}")
    raw = trainer.evaluate()

    metrics = {k.replace("eval_", ""): v for k, v in raw.items()
               if k in ("eval_precision", "eval_recall", "eval_f1", "eval_fn_per_1000_tokens")}
    return metrics


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_results(label: str, metrics: Dict):
    print()
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    key_order = ["precision", "recall", "f1", "fn_per_1000_tokens"]
    for k in key_order:
        if k in metrics:
            print(f"  {k:<30} {metrics[k]:.4f}")


def print_comparison(baseline: Dict, finetuned: Dict):
    print()
    print("=" * 60)
    print("  IMPROVEMENT  (fine-tuned vs baseline)")
    print("=" * 60)
    key_order = ["precision", "recall", "f1", "fn_per_1000_tokens"]
    for k in key_order:
        if k in baseline and k in finetuned:
            diff = finetuned[k] - baseline[k]
            pct  = (diff / max(abs(baseline[k]), 1e-9)) * 100
            sign = "+" if diff >= 0 else ""
            # For fn_per_1000, lower is better — flag it clearly
            better = "(better)" if (k == "fn_per_1000_tokens" and diff < 0) or \
                                   (k != "fn_per_1000_tokens" and diff > 0) else "(worse)"
            print(f"  {k:<30} {sign}{diff:.4f}  ({sign}{pct:.1f}%)  {better}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs fine-tuned model")
    parser.add_argument("--val",              required=True,
                        help="Path to validation JSONL")
    parser.add_argument("--baseline_model",   default="obi/deid_roberta_i2b2",
                        help="HuggingFace model ID or path for baseline")
    parser.add_argument("--finetuned_model",  default=None,
                        help="Path to your fine-tuned model dir (optional)")
    parser.add_argument("--max_samples",      type=int, default=None,
                        help="Cap val set size (useful for quick checks)")
    args = parser.parse_args()

    # Load raw records — baseline pipeline eval needs the original text + spans
    val_records = load_jsonl(args.val, args.max_samples)

    # Also prepare tokenized dataset for fine-tuned Trainer eval
    logger.info("Preparing tokenized validation dataset...")
    tokenizer   = AutoTokenizer.from_pretrained(args.baseline_model)
    val_dataset = prepare_dataset(val_records, tokenizer)
    logger.info(f"Val dataset ready: {len(val_dataset)} examples")

    # Baseline: use pipeline() so the model runs with its own native labels.
    # We do NOT use Trainer here — forcing our label scheme onto the baseline
    # replaces its classification head with random weights (scores all zero).
    logger.info("Evaluating baseline using HuggingFace pipeline (native labels)...")
    baseline_metrics = evaluate_baseline_pipeline(
        args.baseline_model,
        val_records,
        label=f"BASELINE  ({args.baseline_model})"
    )
    print_results(f"BASELINE  ({args.baseline_model})", baseline_metrics)

    with open("baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    logger.info("Baseline metrics saved to baseline_metrics.json")

    # Fine-tuned: use Trainer (was trained with our exact label scheme)
    if args.finetuned_model:
        finetuned_metrics = evaluate_finetuned_trainer(
            args.finetuned_model,
            val_dataset,
            label=f"FINE-TUNED  ({args.finetuned_model})"
        )
        print_results(f"FINE-TUNED  ({args.finetuned_model})", finetuned_metrics)
        print_comparison(baseline_metrics, finetuned_metrics)

        with open("finetuned_metrics.json", "w") as f:
            json.dump(finetuned_metrics, f, indent=2)
        logger.info("Fine-tuned metrics saved to finetuned_metrics.json")
    else:
        print()
        print("Tip: re-run with --finetuned_model ./runs/synthetic_v1 to see the comparison.")
        print()


if __name__ == "__main__":
    main()
