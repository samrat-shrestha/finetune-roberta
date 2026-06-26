"""
loop_retrain.py
---------------
Error-driven retraining loop (loop engineering) for DeIdentiPhi.

This orchestrator iteratively improves the fine-tuned PHI de-identification
model by:
  1. Training the model on the current dataset
  2. Analyzing per-category errors on the validation set
  3. Generating targeted synthetic data emphasizing weak categories
  4. Merging original + targeted data into an augmented training set
  5. Repeating until FN target is met, metrics plateau, or budget is exhausted

Usage:
    # Quick smoke test (CPU, small data, 2 iterations)
    python loop_retrain.py --train data/synthetic_train.jsonl \
                           --val   data/synthetic_val.jsonl   \
                           --max_iterations 2 --max_samples 10 \
                           --epochs 1 --batch_size 2 \
                           --output_dir ./runs/loop_test

    # Full run
    python loop_retrain.py --train data/synthetic_train.jsonl \
                           --val   data/synthetic_val.jsonl   \
                           --max_iterations 5 --fn_target 0.5 \
                           --epochs 3 --batch_size 8 \
                           --output_dir ./runs/loop
"""

import argparse
import json
import logging
import os
import shutil
import time
from typing import Dict, List, Optional

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

# Import from project modules
from error_analysis import analyze_errors, load_jsonl as ea_load_jsonl
from generate_synthetic_data import generate_notes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label system — identical to finetune.py
# ---------------------------------------------------------------------------

BASE_MODEL = "obi/deid_roberta_i2b2"

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
# Shared helpers from finetune.py (kept in sync)
# ---------------------------------------------------------------------------

import re

def char_spans_to_word_bilou(text, spans):
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


def load_jsonl(path, max_samples=None):
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


def prepare_dataset(records, tokenizer, max_length=512):
    all_words, all_labels = [], []
    for record in records:
        pairs = char_spans_to_word_bilou(record["text"], record["spans"])
        all_words.append([w for w, _ in pairs])
        all_labels.append([l for _, l in pairs])

    raw = Dataset.from_dict({"words": all_words, "bilou_labels": all_labels})
    return raw.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, max_length),
        batched=True,
        remove_columns=["words", "bilou_labels"],
    )


def make_compute_metrics(label_list):
    try:
        from seqeval.metrics import f1_score, precision_score, recall_score
        from seqeval.scheme import BILOU
        USE_SEQEVAL = True
    except ImportError:
        logger.warning("seqeval not installed — using token accuracy")
        USE_SEQEVAL = False

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = [[label_list[l] for l in label if l != -100]
                       for label in labels]
        true_preds  = [[label_list[p] for p, l in zip(pred, label) if l != -100]
                       for pred, label in zip(predictions, labels)]

        if USE_SEQEVAL:
            results = {
                "precision": precision_score(true_labels, true_preds, scheme=BILOU, zero_division=0),
                "recall":    recall_score(true_labels, true_preds,    scheme=BILOU, zero_division=0),
                "f1":        f1_score(true_labels, true_preds,        scheme=BILOU, zero_division=0),
            }
        else:
            flat_true = [l for seq in true_labels for l in seq]
            flat_pred = [p for seq in true_preds  for p in seq]
            correct = sum(t == p for t, p in zip(flat_true, flat_pred))
            results = {"accuracy": correct / len(flat_true)}

        fn_count = sum(
            1 for true_seq, pred_seq in zip(true_labels, true_preds)
            for t, p in zip(true_seq, pred_seq)
            if t != "O" and p == "O"
        )
        total_tokens = sum(len(seq) for seq in true_labels)
        results["fn_per_1000_tokens"] = (fn_count / max(total_tokens, 1)) * 1000
        return results

    return compute_metrics


# ---------------------------------------------------------------------------
# Core: single training iteration
# ---------------------------------------------------------------------------

def train_iteration(
    train_records: List[Dict],
    val_records: List[Dict],
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    max_length: int = 512,
    use_lora: bool = False,
) -> Dict:
    """
    Run one full training + evaluation cycle. Returns the evaluation metrics.
    """
    logger.info(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    logger.info("Preparing datasets...")
    train_dataset = prepare_dataset(train_records, tokenizer, max_length)
    val_dataset   = prepare_dataset(val_records, tokenizer, max_length)

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Load model fresh each iteration (from base, not from previous iteration)
    logger.info(f"Loading model from {BASE_MODEL}...")
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Optional LoRA
    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            lora_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=16, lora_alpha=32, lora_dropout=0.1,
                target_modules=["query", "value"],
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            logger.error("peft not installed. Run: pip install peft")
            raise

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(LABEL_LIST),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Training...")
    trainer.train()

    # Save best model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Final evaluation
    logger.info("Evaluating...")
    raw_metrics = trainer.evaluate()
    metrics = {k.replace("eval_", ""): v for k, v in raw_metrics.items()
               if k.startswith("eval_") and k != "eval_runtime"
               and k != "eval_samples_per_second"
               and k != "eval_steps_per_second"}

    return metrics


# ---------------------------------------------------------------------------
# Core: merge training data
# ---------------------------------------------------------------------------

def merge_jsonl_files(paths: List[str], output_path: str) -> int:
    """Merge multiple JSONL files into one. Returns total record count."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    count = 0
    with open(output_path, "w") as out:
        for path in paths:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        count += 1
    logger.info(f"Merged {len(paths)} files → {output_path} ({count} records)")
    return count


# ---------------------------------------------------------------------------
# The Loop
# ---------------------------------------------------------------------------

def run_loop(args):
    """Main loop: train → analyze → generate targeted data → merge → repeat."""

    os.makedirs(args.output_dir, exist_ok=True)

    # Track metrics across iterations
    history = []
    best_fn_per_1000 = float("inf")
    best_iter = -1

    # Current training data path — starts as the original
    current_train_path = args.train

    print()
    print("=" * 70)
    print("  ERROR-DRIVEN RETRAINING LOOP")
    print("=" * 70)
    print(f"  Max iterations:  {args.max_iterations}")
    print(f"  FN target:       {args.fn_target} per 1000 tokens")
    print(f"  F1 floor:        {args.f1_floor} (model must exceed this to be valid)")
    print(f"  Base train data: {args.train}")
    print(f"  Validation data: {args.val}")
    print(f"  Output:          {args.output_dir}")
    print("=" * 70)
    print()

    for iteration in range(1, args.max_iterations + 1):
        iter_dir = os.path.join(args.output_dir, f"iter_{iteration}")
        model_dir = os.path.join(iter_dir, "model")
        os.makedirs(iter_dir, exist_ok=True)

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  ITERATION {iteration} / {args.max_iterations}")
        logger.info("=" * 70)

        # --- Step 1: Load training data ---
        train_records = load_jsonl(current_train_path, args.max_samples)
        val_records   = load_jsonl(args.val, args.max_samples)

        # --- Step 2: Train ---
        logger.info(f"[Iter {iteration}] Training on {len(train_records)} records...")
        start_time = time.time()
        metrics = train_iteration(
            train_records=train_records,
            val_records=val_records,
            output_dir=model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            use_lora=args.use_lora,
        )
        train_time = time.time() - start_time

        # --- Step 3: Error analysis ---
        logger.info(f"[Iter {iteration}] Running error analysis...")
        error_report = analyze_errors(
            model_path=model_dir,
            val_records=val_records,
            max_samples=args.max_samples,
        )

        # Save error report
        report_path = os.path.join(iter_dir, "error_report.json")
        with open(report_path, "w") as f:
            json.dump(error_report, f, indent=2)

        # Use error analysis FN rate (more accurate since it uses pipeline)
        fn_per_1000 = error_report["overall"]["fn_per_1000"]
        fp_per_1000 = error_report["overall"]["fp_per_1000"]
        metrics["fn_per_1000_from_analysis"] = fn_per_1000
        metrics["fp_per_1000_from_analysis"] = fp_per_1000

        # Save iteration metrics
        iter_result = {
            "iteration": iteration,
            "train_records": len(train_records),
            "train_time_seconds": round(train_time, 1),
            "metrics": metrics,
            "fn_per_1000_analysis": fn_per_1000,
            "category_weights": error_report["category_weights"],
            "per_category_fn": {
                cat: error_report["per_category"][cat]["fn"]
                for cat in PHI_CATEGORIES
                if error_report["per_category"][cat]["fn"] > 0
            },
        }
        history.append(iter_result)

        # Save metrics for this iteration
        metrics_path = os.path.join(iter_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(iter_result, f, indent=2)

        # Print iteration summary
        print()
        print(f"  ┌─ Iteration {iteration} Results ─────────────────────────────")
        print(f"  │  Train records:    {len(train_records)}")
        print(f"  │  Train time:       {train_time:.0f}s")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  │  {k:<25} {v:.4f}")
        print(f"  │  FN/1000 (analysis): {fn_per_1000:.4f}")
        print(f"  │  FP/1000 (analysis): {fp_per_1000:.4f}")
        if error_report["per_category"]:
            top_fn_cats = sorted(
                [(cat, error_report["per_category"][cat]["fn"])
                 for cat in PHI_CATEGORIES
                 if error_report["per_category"][cat]["fn"] > 0],
                key=lambda x: -x[1]
            )
            if top_fn_cats:
                print(f"  │  Top FN categories: {', '.join(f'{c}({n})' for c, n in top_fn_cats[:5])}")
        print(f"  └───────────────────────────────────────────────────")
        print()

        # Get F1 from trainer metrics (use 0.0 if not available)
        current_f1 = metrics.get("f1", 0.0)
        model_is_valid = current_f1 >= args.f1_floor

        # Track best — only consider models that pass the F1 floor
        if model_is_valid and fn_per_1000 < best_fn_per_1000:
            best_fn_per_1000 = fn_per_1000
            best_iter = iteration
            # Copy best model
            best_dir = os.path.join(args.output_dir, "best_model")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(model_dir, best_dir)
            logger.info(f"[Iter {iteration}] New best! FN/1000 = {fn_per_1000:.4f} (F1 = {current_f1:.4f})")
        elif not model_is_valid:
            logger.warning(
                f"[Iter {iteration}] Model below F1 floor: F1 = {current_f1:.4f} < {args.f1_floor}. "
                f"FN/1000 = {fn_per_1000:.4f} is unreliable (model may be over/under-predicting). "
                f"Continuing to next iteration."
            )

        # --- Step 4: Check exit conditions ---

        # 4a: Target reached — BUT only if model is actually good (F1 above floor)
        # A model that predicts everything as PHI gets 0 FN but is useless.
        if fn_per_1000 <= args.fn_target and model_is_valid:
            logger.info(f"✅ FN target reached! FN/1000 = {fn_per_1000:.4f} ≤ {args.fn_target}, F1 = {current_f1:.4f} ≥ {args.f1_floor}")
            print(f"\n  ✅ TARGET REACHED at iteration {iteration}!")
            print(f"     FN/1000 = {fn_per_1000:.4f} (target: {args.fn_target})")
            print(f"     F1      = {current_f1:.4f} (floor: {args.f1_floor})")
            break
        elif fn_per_1000 <= args.fn_target and not model_is_valid:
            logger.info(
                f"⚠️ FN target met ({fn_per_1000:.4f}) but F1 too low ({current_f1:.4f} < {args.f1_floor}). "
                f"Model is likely over-predicting. Continuing..."
            )
            print(f"\n  ⚠️  FN target met but model unreliable (F1 = {current_f1:.4f} < {args.f1_floor})")
            print(f"     The model is likely predicting everything as PHI. Continuing...")

        # 4b: Plateau detection (no improvement for patience iterations)
        if len(history) >= 2:
            prev_fn = history[-2]["fn_per_1000_analysis"]
            improvement = prev_fn - fn_per_1000
            improvement_pct = (improvement / max(prev_fn, 1e-9)) * 100

            if improvement <= 0:
                logger.info(f"⛔ No improvement: {prev_fn:.4f} → {fn_per_1000:.4f}")
                print(f"\n  ⛔ PLATEAU at iteration {iteration}")
                print(f"     FN/1000: {prev_fn:.4f} → {fn_per_1000:.4f} (no improvement)")
                print(f"     Best was iteration {best_iter} with FN/1000 = {best_fn_per_1000:.4f}")
                break
            else:
                logger.info(f"📈 Improved: {prev_fn:.4f} → {fn_per_1000:.4f} ({improvement_pct:+.1f}%)")

        # 4c: Last iteration
        if iteration == args.max_iterations:
            logger.info(f"⏱️ Max iterations ({args.max_iterations}) reached")
            print(f"\n  ⏱️ MAX ITERATIONS reached")
            print(f"     Best was iteration {best_iter} with FN/1000 = {best_fn_per_1000:.4f}")
            break

        # --- Step 5: Generate targeted data ---
        logger.info(f"[Iter {iteration}] Generating targeted training data...")
        targeted_path = os.path.join(iter_dir, "targeted_data.jsonl")
        targeted_notes = args.targeted_notes_per_iter

        generate_notes(
            num_notes=targeted_notes,
            output_path=targeted_path,
            seed=42 + iteration * 1000,  # different seed each iteration
            category_weights=error_report["category_weights"],
        )

        # --- Step 6: Merge original + targeted data ---
        merged_path = os.path.join(iter_dir, "merged_train.jsonl")
        files_to_merge = [args.train, targeted_path]

        # Also include targeted data from previous iterations (accumulate)
        for prev_iter in range(1, iteration):
            prev_targeted = os.path.join(args.output_dir, f"iter_{prev_iter}", "targeted_data.jsonl")
            if os.path.exists(prev_targeted):
                files_to_merge.append(prev_targeted)

        merge_jsonl_files(files_to_merge, merged_path)
        current_train_path = merged_path

    # --- Save summary ---
    summary = {
        "total_iterations": len(history),
        "best_iteration": best_iter,
        "best_fn_per_1000": best_fn_per_1000,
        "fn_target": args.fn_target,
        "target_reached": best_fn_per_1000 <= args.fn_target,
        "history": history,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print()
    print("=" * 70)
    print("  LOOP SUMMARY")
    print("=" * 70)
    print(f"  Total iterations:  {len(history)}")
    print(f"  Best iteration:    {best_iter}")
    print(f"  Best FN/1000:      {best_fn_per_1000:.4f}")
    print(f"  Target:            {args.fn_target}")
    print(f"  Target reached:    {'✅ YES' if summary['target_reached'] else '❌ NO'}")
    print()
    print("  Iteration trajectory:")
    for h in history:
        marker = " ★" if h["iteration"] == best_iter else ""
        print(f"    Iter {h['iteration']}: FN/1000 = {h['fn_per_1000_analysis']:.4f} "
              f"(train: {h['train_records']} records, {h['train_time_seconds']}s){marker}")
    print()
    print(f"  Best model saved to: {os.path.join(args.output_dir, 'best_model')}")
    print(f"  Full summary:        {summary_path}")
    print("=" * 70)
    print()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Error-driven retraining loop for PHI de-identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test
  python loop_retrain.py --train data/synthetic_train.jsonl \\
                         --val data/synthetic_val.jsonl \\
                         --max_iterations 2 --max_samples 10 \\
                         --epochs 1 --batch_size 2 \\
                         --output_dir ./runs/loop_test

  # Full run
  python loop_retrain.py --train data/synthetic_train.jsonl \\
                         --val data/synthetic_val.jsonl \\
                         --max_iterations 5 --fn_target 0.5 \\
                         --output_dir ./runs/loop
        """,
    )

    # Data
    parser.add_argument("--train",       required=True, help="Path to training JSONL")
    parser.add_argument("--val",         required=True, help="Path to validation JSONL")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap dataset size (for smoke tests)")

    # Loop control
    parser.add_argument("--max_iterations", type=int, default=5,
                        help="Maximum number of loop iterations (default: 5)")
    parser.add_argument("--fn_target",      type=float, default=0.5,
                        help="Stop when FN/1000 tokens drops below this (default: 0.5)")
    parser.add_argument("--f1_floor",       type=float, default=0.8,
                        help="Minimum F1 required for a model to be considered valid (default: 0.8). "
                             "Prevents undertrained models from gaming the FN metric.")
    parser.add_argument("--targeted_notes_per_iter", type=int, default=200,
                        help="Number of targeted notes to generate per iteration (default: 200)")

    # Training hyperparameters
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--max_length",  type=int,   default=512)
    parser.add_argument("--use_lora",    action="store_true")

    # Output
    parser.add_argument("--output_dir",  default="./runs/loop")

    args = parser.parse_args()
    run_loop(args)


if __name__ == "__main__":
    main()
