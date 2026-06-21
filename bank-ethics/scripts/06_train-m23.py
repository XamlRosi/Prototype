#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M2/M3: Fine-tuned multilingual transformer for Responsible AI multi-label classification.

M2:
  python scripts/06_train-m23.py \
    --csv data/training_dataset.csv \
    --model-id M2 \
    --model-name bert-base-multilingual-cased \
    --outdir data/models/m2_mbert

M3:
  python scripts/06_train-m23.py \
    --csv data/training_dataset.csv \
    --model-id M3 \
    --model-name xlm-roberta-base \
    --outdir data/models/m3_xlm_roberta
"""

import os
import json
import argparse
import inspect
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_TARGETS = [
    "unsafe",
    "privacy_violation",
    "bias",
    "manipulation",
    "transparency_issue",
    "honesty_issue",
    "policy_issue",
    "accessibility_issue",
    "missing_human_escalation",
]


def make_text(df: pd.DataFrame, text_mode: str) -> List[str]:
    if text_mode == "metadata":
        if "text_with_metadata" in df.columns:
            return df["text_with_metadata"].fillna("").astype(str).tolist()
        text_mode = "separate"

    if text_mode == "combined":
        if "text" not in df.columns:
            raise ValueError("CSV is missing 'text' column.")
        return df["text"].fillna("").astype(str).tolist()

    if text_mode == "separate":
        if "prompt" in df.columns and "answer" in df.columns:
            return (
                "USER: " + df["prompt"].fillna("").astype(str)
                + "\nASSISTANT: " + df["answer"].fillna("").astype(str)
            ).tolist()
        if "text" in df.columns:
            return df["text"].fillna("").astype(str).tolist()

    raise ValueError("CSV must contain text_with_metadata, text, or prompt+answer.")


def add_derived_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Positive class = issue / risk / non-compliance.
    """

    if "transparency_issue_bin" not in df.columns and "transparency_score" in df.columns:
        df["transparency_issue_bin"] = (
            df["transparency_score"].fillna(0).astype(int) < 2
        ).astype(int)

    if "honesty_issue_bin" not in df.columns and "honesty_score" in df.columns:
        df["honesty_issue_bin"] = (
            df["honesty_score"].fillna(0).astype(int) < 2
        ).astype(int)

    if "policy_issue_bin" not in df.columns and "policy_alignment" in df.columns:
        df["policy_issue_bin"] = (
            df["policy_alignment"].fillna(0).astype(int) < 2
        ).astype(int)

    if "accessibility_issue_bin" not in df.columns and "accessibility_score" in df.columns:
        df["accessibility_issue_bin"] = (
            df["accessibility_score"].fillna(0).astype(int) < 2
        ).astype(int)

    return df


def get_target_array(df: pd.DataFrame, target: str) -> np.ndarray:
    bin_col = f"{target}_bin"

    if bin_col in df.columns:
        return df[bin_col].fillna(0).astype(int).values

    if target in df.columns:
        return (df[target].fillna(0).astype(int) > 0).astype(int).values

    raise ValueError(f"Missing target column: {target} or {bin_col}")


def filter_valid_targets(df: pd.DataFrame, targets: List[str]):
    active = []
    skipped = {}

    for target in targets:
        try:
            y = get_target_array(df, target)
        except Exception as e:
            skipped[target] = f"missing: {e}"
            continue

        if len(np.unique(y)) < 2:
            skipped[target] = f"constant target: only class {int(np.unique(y)[0])}"
        else:
            active.append(target)

    return active, skipped


def build_groups(df: pd.DataFrame) -> Optional[np.ndarray]:
    possible_cols = ["pair_id", "conversation_id", "scenario_id", "prompt_id"]

    groups = []
    has_any_group = False

    for idx, row in df.iterrows():
        group_value = None
        group_col = None

        for col in possible_cols:
            if col in df.columns:
                value = row.get(col)
                if pd.notna(value) and str(value).strip():
                    group_value = str(value).strip()
                    group_col = col
                    has_any_group = True
                    break

        if group_value is None:
            groups.append(f"row:{idx}")
        else:
            groups.append(f"{group_col}:{group_value}")

    if not has_any_group:
        return None

    return np.array(groups)


def grouped_split_indices(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
):
    indices = np.arange(len(df))
    groups = build_groups(df)

    if groups is not None and len(np.unique(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(indices, groups=groups))
        return train_idx, test_idx, "GroupShuffleSplit"

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    return train_idx, test_idx, "Random train_test_split"


def make_train_val_split(
    train_idx: np.ndarray,
    df: pd.DataFrame,
    val_size: float,
    seed: int,
):
    train_df = df.iloc[train_idx].copy()
    local_indices = np.arange(len(train_df))
    groups = build_groups(train_df)

    if groups is not None and len(np.unique(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        local_train, local_val = next(splitter.split(local_indices, groups=groups))
    else:
        local_train, local_val = train_test_split(
            local_indices,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
        )

    return train_idx[local_train], train_idx[local_val]


class MultiLabelTextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int,
    ):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )

        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def detailed_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    targets: List[str],
    threshold: float,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_label = {}

    for i, target in enumerate(targets):
        tn, fp, fn, tp = confusion_matrix(
            y_true[:, i],
            y_pred[:, i],
            labels=[0, 1],
        ).ravel()

        per_label[target] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    return {
        "threshold": threshold,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_label": per_label,
    }


def compute_metrics_builder(threshold: float):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = sigmoid_np(logits)
        preds = (probs >= threshold).astype(int)

        return {
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        }

    return compute_metrics


def make_training_args(args):
    """
    Compatible with transformers versions using either evaluation_strategy
    or eval_strategy.
    """

    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    kwargs = {
        "output_dir": args.outdir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "logging_steps": 20,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "seed": args.seed,
        "report_to": "none",
    }

    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"

    if "save_total_limit" in params:
        kwargs["save_total_limit"] = 2

    return TrainingArguments(**kwargs)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True)
    ap.add_argument("--model-id", required=True, choices=["M2", "M3"])
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--text-mode", choices=["metadata", "combined", "separate"], default="metadata")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))

    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.5)

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = add_derived_targets(df)

    requested_targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    active_targets, skipped_targets = filter_valid_targets(df, requested_targets)

    if not active_targets:
        raise ValueError("No valid targets with both positive and negative classes.")

    texts = make_text(df, args.text_mode)
    Y = np.stack([get_target_array(df, t) for t in active_targets], axis=1)

    train_idx, test_idx, split_type = grouped_split_indices(
        df=df,
        test_size=args.test_size,
        seed=args.seed,
    )

    train_idx, val_idx = make_train_val_split(
        train_idx=train_idx,
        df=df,
        val_size=args.val_size,
        seed=args.seed,
    )

    print("=" * 80)
    print(f"{args.model_id}: Fine-tuned transformer")
    print("=" * 80)
    print(f"Model name: {args.model_name}")
    print(f"Rows: {len(df)}")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"Split: {split_type}")
    print(f"Active targets: {active_targets}")
    print(f"Skipped targets: {skipped_targets}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(active_targets),
        problem_type="multi_label_classification",
    )

    train_dataset = MultiLabelTextDataset(
        texts=[texts[i] for i in train_idx],
        labels=Y[train_idx],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    val_dataset = MultiLabelTextDataset(
        texts=[texts[i] for i in val_idx],
        labels=Y[val_idx],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    test_dataset = MultiLabelTextDataset(
        texts=[texts[i] for i in test_idx],
        labels=Y[test_idx],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    training_args = make_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_builder(args.threshold),
    )

    trainer.train()

    test_output = trainer.predict(test_dataset)
    test_logits = test_output.predictions
    test_probs = sigmoid_np(test_logits)

    report = detailed_metrics(
        y_true=Y[test_idx],
        y_prob=test_probs,
        targets=active_targets,
        threshold=args.threshold,
    )

    final_report = {
        "model_id": args.model_id,
        "model_name": args.model_name,
        "dataset_csv": args.csv,
        "text_mode": args.text_mode,
        "split_type": split_type,
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "threshold": args.threshold,
        "requested_targets": requested_targets,
        "active_targets": active_targets,
        "skipped_targets": skipped_targets,
        "metrics": report,
    }

    report_path = os.path.join(args.outdir, f"{args.model_id.lower()}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    test_pred_df = df.iloc[test_idx].copy()

    y_pred = (test_probs >= args.threshold).astype(int)

    for i, target in enumerate(active_targets):
        test_pred_df[f"true_{target}"] = Y[test_idx, i]
        test_pred_df[f"prob_{target}"] = test_probs[:, i]
        test_pred_df[f"pred_{target}"] = y_pred[:, i]

    pred_path = os.path.join(args.outdir, f"{args.model_id.lower()}_test_predictions.csv")
    test_pred_df.to_csv(pred_path, index=False)

    final_model_dir = os.path.join(args.outdir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    metadata = {
        "model_id": args.model_id,
        "model_name": args.model_name,
        "targets": active_targets,
        "text_mode": args.text_mode,
        "threshold": args.threshold,
    }

    with open(os.path.join(final_model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(f"- {report_path}")
    print(f"- {pred_path}")
    print(f"- {final_model_dir}")

    print("\nMain metrics:")
    print(f"Macro-F1: {report['macro_f1']:.4f}")
    print(f"Micro-F1: {report['micro_f1']:.4f}")

    for target, m in report["per_label"].items():
        print(
            f"{target:28s} "
            f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
            f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}"
        )


if __name__ == "__main__":
    main()