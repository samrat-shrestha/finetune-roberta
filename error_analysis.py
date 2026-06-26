"""
error_analysis.py
-----------------
Runs a fine-tuned model on the validation set and produces a per-category
error breakdown — the key input for the error-driven retraining loop.

For each PHI category (PATIENT, DATE, LOC, AGE, etc.), it counts:
  - False Negatives (FN): PHI tokens the model missed (predicted as O)
  - False Positives (FP): non-PHI tokens the model incorrectly flagged
  - Total ground-truth PHI tokens
  - FN rate = FN / total

The output includes `category_weights` — higher weights for categories with
more false negatives. These weights drive the targeted data generation step.

Usage (standalone):
    python error_analysis.py --model ./runs/synthetic_v1 \
                             --val data/synthetic_val.jsonl \
                             --output error_report.json

Usage (as importable module):
    from error_analysis import analyze_errors
    report = analyze_errors(model_path, val_records)
"""

import argparse
import json
import logging
import re
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label system — identical to finetune.py
# ---------------------------------------------------------------------------

PHI_CATEGORIES = [
    "PATIENT", "STAFF", "DATE", "LOC", "HOSP",
    "AGE", "ID", "PHONE", "PATORG", "EMAIL", "OTHERPHI",
]

LABEL_LIST = ["O"]
for cat in PHI_CATEGORIES:
    LABEL_LIST += [f"B-{cat}", f"I-{cat}", f"L-{cat}", f"U-{cat}"]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Shared helpers (kept in sync with finetune.py / baseline_eval.py)
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
    """Convert character-level PHI spans to (word, bilou_label) pairs."""
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


def _extract_category(label: str) -> Optional[str]:
    """Extract the PHI category from a BILOU label like 'B-PATIENT' → 'PATIENT'."""
    if label == "O":
        return None
    parts = label.split("-", 1)
    return parts[1] if len(parts) == 2 else None


# ---------------------------------------------------------------------------
# Core: per-category error analysis
# ---------------------------------------------------------------------------

def analyze_errors(
    model_path: str,
    val_records: List[Dict],
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Run the fine-tuned model on validation records and produce a per-category
    error breakdown.

    Returns a dict with:
      - "overall": aggregate FN/FP/total stats
      - "per_category": per-PHI-category breakdown
      - "category_weights": computed weights for targeted data generation
      - "fn_examples": sample false-negative words for debugging
    """
    from transformers import pipeline as hf_pipeline

    if max_samples:
        val_records = val_records[:max_samples]

    logger.info(f"Loading model for error analysis: {model_path}")
    pipe = hf_pipeline(
        "token-classification",
        model=model_path,
        aggregation_strategy="none",
        device=-1,  # CPU — safe default
    )

    # Per-category counters
    cat_fn = defaultdict(int)       # false negatives by category
    cat_fp = defaultdict(int)       # false positives by category
    cat_total = defaultdict(int)    # total ground-truth PHI tokens by category
    total_tokens = 0
    total_fn = 0
    total_fp = 0

    # Collect example FN words for debugging
    fn_examples = defaultdict(list)
    MAX_EXAMPLES = 10  # cap per category

    for rec_idx, record in enumerate(val_records):
        text  = record["text"]
        spans = record["spans"]

        # Ground truth
        word_label_pairs = char_spans_to_word_bilou(text, spans)
        true_labels = [l for _, l in word_label_pairs]

        # Prediction
        try:
            raw_preds = pipe(text)
        except Exception as e:
            logger.warning(f"Pipeline failed on record {rec_idx}: {e}")
            continue

        # Build char → predicted label mapping
        char_to_pred = {}
        for tok in raw_preds:
            for c in range(tok["start"], tok["end"]):
                char_to_pred[c] = tok["entity"]

        # Align predictions to words
        pred_labels = []
        for match in re.finditer(r'\S+', text):
            word_start = match.start()
            word_end   = match.end()

            preds_in_word = [char_to_pred.get(c, "O") for c in range(word_start, word_end)]
            non_o = [p for p in preds_in_word if p != "O"]
            if non_o:
                raw_label = max(set(non_o), key=non_o.count)
            else:
                raw_label = "O"

            # Normalize: keep prefix-CATEGORY format
            if raw_label == "O":
                pred_labels.append("O")
            else:
                parts = raw_label.split("-", 1)
                prefix   = parts[0] if len(parts) == 2 else "B"
                category = parts[1] if len(parts) == 2 else parts[0]
                pred_labels.append(f"{prefix}-{category}")

        # Compare — word by word
        words = [w for w, _ in word_label_pairs]
        min_len = min(len(true_labels), len(pred_labels))

        for i in range(min_len):
            total_tokens += 1
            true_cat = _extract_category(true_labels[i])
            pred_cat = _extract_category(pred_labels[i])

            if true_cat:
                cat_total[true_cat] += 1

            # False negative: true is PHI, predicted is O
            if true_cat and not pred_cat:
                total_fn += 1
                cat_fn[true_cat] += 1
                if len(fn_examples[true_cat]) < MAX_EXAMPLES:
                    fn_examples[true_cat].append({
                        "word": words[i],
                        "true_label": true_labels[i],
                        "record_idx": rec_idx,
                    })

            # False positive: true is O, predicted is PHI
            if not true_cat and pred_cat:
                total_fp += 1
                cat_fp[pred_cat] += 1

    # --- Build the report ---
    overall = {
        "fn": total_fn,
        "fp": total_fp,
        "total_phi_tokens": sum(cat_total.values()),
        "total_tokens": total_tokens,
        "fn_per_1000": (total_fn / max(total_tokens, 1)) * 1000,
        "fp_per_1000": (total_fp / max(total_tokens, 1)) * 1000,
    }

    per_category = {}
    for cat in PHI_CATEGORIES:
        total = cat_total.get(cat, 0)
        fn    = cat_fn.get(cat, 0)
        fp    = cat_fp.get(cat, 0)
        per_category[cat] = {
            "fn": fn,
            "fp": fp,
            "total": total,
            "fn_rate": fn / max(total, 1),
        }

    # --- Compute category weights ---
    # Strategy: weight = 1.0 + (category_fn_rate / max_fn_rate) * boost_factor
    # Categories with higher FN rates get higher weights
    fn_rates = {cat: per_category[cat]["fn_rate"] for cat in PHI_CATEGORIES}
    max_fn_rate = max(fn_rates.values()) if fn_rates else 0.0
    BOOST_FACTOR = 3.0  # how much extra emphasis to give the worst category

    category_weights = {}
    for cat in PHI_CATEGORIES:
        if max_fn_rate > 0 and cat_total.get(cat, 0) > 0:
            normalized = fn_rates[cat] / max_fn_rate
            category_weights[cat] = round(1.0 + normalized * BOOST_FACTOR, 2)
        else:
            category_weights[cat] = 1.0

    report = {
        "overall": overall,
        "per_category": per_category,
        "category_weights": category_weights,
        "fn_examples": {cat: examples for cat, examples in fn_examples.items()},
    }

    # Log summary
    logger.info("=" * 60)
    logger.info("ERROR ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total tokens analyzed: {total_tokens}")
    logger.info(f"  Total FN: {total_fn}  ({overall['fn_per_1000']:.2f} per 1000 tokens)")
    logger.info(f"  Total FP: {total_fp}  ({overall['fp_per_1000']:.2f} per 1000 tokens)")
    logger.info("")
    logger.info(f"  {'Category':<12} {'FN':>4} {'FP':>4} {'Total':>6} {'FN Rate':>8} {'Weight':>7}")
    logger.info(f"  {'-'*12} {'-'*4} {'-'*4} {'-'*6} {'-'*8} {'-'*7}")
    for cat in PHI_CATEGORIES:
        c = per_category[cat]
        w = category_weights[cat]
        if c["total"] > 0:
            logger.info(
                f"  {cat:<12} {c['fn']:>4} {c['fp']:>4} {c['total']:>6} "
                f"{c['fn_rate']:>8.4f} {w:>7.2f}"
            )
    logger.info("=" * 60)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-category error analysis for PHI de-identification")
    parser.add_argument("--model",       required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--val",         required=True, help="Path to validation JSONL")
    parser.add_argument("--output",      default="error_report.json", help="Output JSON report path")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap validation set size")
    args = parser.parse_args()

    val_records = load_jsonl(args.val, args.max_samples)
    report = analyze_errors(args.model, val_records)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
